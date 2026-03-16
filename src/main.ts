import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * Detects geometric shapes directly from raw canvas pixels.
   * Pipeline: grayscale -> Sobel edges -> threshold/mask -> connected components
   * -> contour/hull features -> rule+score based classification.
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    const { width, height } = imageData;
    // 1) Convert pixels to analysis-friendly channels.
    const grayscale = this.toGrayscale(imageData);
    const edges = this.computeSobelMagnitude(grayscale, width, height);

    // 2) Build a clean foreground mask that is robust to background noise.
    const mask = this.createForegroundMask(grayscale, edges, width, height);

    // 3) Extract connected blobs that can represent candidate shapes.
    const minArea = Math.max(90, Math.floor((width * height) * 0.001));
    const components = this.findConnectedComponents(mask, width, height, minArea);

    const shapes: DetectedShape[] = [];
    for (const component of components) {
      const compactness =
        component.perimeter > 0
          ? component.area / (component.perimeter * component.perimeter)
          : 0;

      // Reject thin/noisy blobs (line fragments and text strokes).
      if (component.area < 180 && compactness < 0.018) {
        continue;
      }

      const center: Point = {
        x: component.sumX / component.area,
        y: component.sumY / component.area,
      };

      const boundaryPoints = component.boundary.map((index) => ({
        x: index % width,
        y: Math.floor(index / width),
      }));

      const orderedContour = this.orderContourByAngle(boundaryPoints, center);
      if (orderedContour.length < 3) {
        continue;
      }

      const epsilon = Math.max(1.5, component.perimeter * 0.018);
      const simplified = this.simplifyContour(orderedContour, epsilon);
      if (simplified.length < 3) {
        continue;
      }

      const polygon = this.removeNearCollinear(simplified);
      if (polygon.length < 3) {
        continue;
      }

      const convexHull = this.computeConvexHull(boundaryPoints);
      const hullPerimeter = this.computeClosedPerimeter(convexHull);
      const hullPolygon = this.simplifyClosedPolygon(
        convexHull,
        Math.max(1.8, hullPerimeter * 0.02)
      );
      const convexHullArea = Math.max(1, Math.abs(this.computePolygonArea(convexHull)));
      const circularity =
        component.perimeter > 0
          ? (4 * Math.PI * component.area) / (component.perimeter * component.perimeter)
          : 0;

      const bboxWidth = component.bbox.maxX - component.bbox.minX + 1;
      const bboxHeight = component.bbox.maxY - component.bbox.minY + 1;
      const aspectRatio = bboxHeight > 0 ? bboxWidth / bboxHeight : 1;
      const extent = component.area / Math.max(1, bboxWidth * bboxHeight);

      if (component.area < 500 && extent < 0.22) {
        continue;
      }

      const concavityRatio = this.calculateConcavityRatio(polygon);
      const rectangleAngleScore = this.computeRectangleAngleScore(
        hullPolygon.length === 4 ? hullPolygon : polygon
      );
      const solidity = component.area / convexHullArea;
      const radial = this.computeRadialFeatures(boundaryPoints, center);

      // 4) Classify each blob using geometric features and tuned decision rules.
      const classification = this.classifyShape({
        vertexCount: polygon.length,
        convexVertexCount: Math.max(3, hullPolygon.length),
        circularity,
        aspectRatio,
        extent,
        solidity,
        concavityRatio,
        radialVariation: radial.coefficientOfVariation,
        radialPeaks: radial.peakCount,
        compactness,
        area: component.area,
        rectangleAngleScore,
      });

      shapes.push({
        type: classification.type,
        confidence: classification.confidence,
        boundingBox: {
          x: component.bbox.minX,
          y: component.bbox.minY,
          width: component.bbox.maxX - component.bbox.minX + 1,
          height: component.bbox.maxY - component.bbox.minY + 1,
        },
        center,
        area: component.area,
      });
    }

    // 5) Return detections sorted by confidence for stable downstream evaluation.
    shapes.sort((a, b) => b.confidence - a.confidence || b.area - a.area);

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  private toGrayscale(imageData: ImageData): Float32Array {
    const { data, width, height } = imageData;
    const grayscale = new Float32Array(width * height);

    for (let i = 0, p = 0; p < grayscale.length; i += 4, p++) {
      // Perceptual luminance approximation.
      grayscale[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    return grayscale;
  }

  private computeSobelMagnitude(
    grayscale: Float32Array,
    width: number,
    height: number
  ): Float32Array {
    const result = new Float32Array(width * height);
    const index = (x: number, y: number) => y * width + x;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const p00 = grayscale[index(x - 1, y - 1)];
        const p01 = grayscale[index(x, y - 1)];
        const p02 = grayscale[index(x + 1, y - 1)];
        const p10 = grayscale[index(x - 1, y)];
        const p12 = grayscale[index(x + 1, y)];
        const p20 = grayscale[index(x - 1, y + 1)];
        const p21 = grayscale[index(x, y + 1)];
        const p22 = grayscale[index(x + 1, y + 1)];

        const gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
        const gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

        result[index(x, y)] = Math.hypot(gx, gy);
      }
    }

    return result;
  }

  private createForegroundMask(
    grayscale: Float32Array,
    edges: Float32Array,
    width: number,
    height: number
  ): Uint8Array {
    const mask = new Uint8Array(width * height);
    const intensityThreshold = this.computeIntensityThreshold(grayscale);
    const edgeThreshold = this.computeEdgeThreshold(edges);

    for (let i = 0; i < mask.length; i++) {
      const darkStrong = grayscale[i] <= intensityThreshold;
      const darkMedium = grayscale[i] <= intensityThreshold + 10;
      const strongEdge = edges[i] >= edgeThreshold;
      // Allow edge support only around already-dark regions.
      mask[i] = darkStrong || (darkMedium && strongEdge) ? 1 : 0;
    }

    // Lightweight morphology to suppress isolated noise and close tiny gaps.
    this.majorityFilter(mask, width, height, 5);
    this.majorityFilter(mask, width, height, 4);

    return mask;
  }

  private computeIntensityThreshold(grayscale: Float32Array): number {
    // Otsu gives robust separation between dark shapes and light backgrounds.
    const histogram = new Uint32Array(256);
    for (let i = 0; i < grayscale.length; i++) {
      const value = Math.max(0, Math.min(255, Math.round(grayscale[i])));
      histogram[value]++;
    }

    const total = grayscale.length;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
      sum += i * histogram[i];
    }

    let sumBackground = 0;
    let weightBackground = 0;
    let maxVariance = -1;
    let bestThreshold = 127;

    for (let i = 0; i < 256; i++) {
      weightBackground += histogram[i];
      if (weightBackground === 0) continue;

      const weightForeground = total - weightBackground;
      if (weightForeground === 0) break;

      sumBackground += i * histogram[i];
      const meanBackground = sumBackground / weightBackground;
      const meanForeground = (sum - sumBackground) / weightForeground;

      const betweenClassVariance =
        weightBackground *
        weightForeground *
        (meanBackground - meanForeground) *
        (meanBackground - meanForeground);

      if (betweenClassVariance > maxVariance) {
        maxVariance = betweenClassVariance;
        bestThreshold = i;
      }
    }

    // Clamp to avoid pulling in gray noise/background textures.
    return Math.max(35, Math.min(120, bestThreshold + 8));
  }

  private computeEdgeThreshold(edges: Float32Array): number {
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < edges.length; i++) {
      const v = edges[i];
      sum += v;
      sumSq += v * v;
    }

    const mean = sum / edges.length;
    const variance = Math.max(0, sumSq / edges.length - mean * mean);
    const std = Math.sqrt(variance);
    return mean + std * 1.7;
  }

  private majorityFilter(
    mask: Uint8Array,
    width: number,
    height: number,
    threshold: number
  ): void {
    const copy = new Uint8Array(mask);
    const index = (x: number, y: number) => y * width + x;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            count += copy[index(x + dx, y + dy)];
          }
        }
        mask[index(x, y)] = count >= threshold ? 1 : 0;
      }
    }
  }

  private findConnectedComponents(
    mask: Uint8Array,
    width: number,
    height: number,
    minArea: number
  ): Array<{
    area: number;
    sumX: number;
    sumY: number;
    boundary: number[];
    perimeter: number;
    bbox: { minX: number; minY: number; maxX: number; maxY: number };
  }> {
    const visited = new Uint8Array(mask.length);
    const components: Array<{
      area: number;
      sumX: number;
      sumY: number;
      boundary: number[];
      perimeter: number;
      bbox: { minX: number; minY: number; maxX: number; maxY: number };
    }> = [];

    const neighbors8 = [
      [-1, -1],
      [0, -1],
      [1, -1],
      [-1, 0],
      [1, 0],
      [-1, 1],
      [0, 1],
      [1, 1],
    ] as const;

    const neighbors4 = [
      [0, -1],
      [-1, 0],
      [1, 0],
      [0, 1],
    ] as const;

    for (let start = 0; start < mask.length; start++) {
      if (mask[start] === 0 || visited[start]) continue;

      const queue: number[] = [start];
      visited[start] = 1;
      let head = 0;

      let area = 0;
      let sumX = 0;
      let sumY = 0;
      let minX = width;
      let minY = height;
      let maxX = 0;
      let maxY = 0;
      let perimeter = 0;
      const boundary: number[] = [];

      while (head < queue.length) {
        const current = queue[head++];
        const x = current % width;
        const y = Math.floor(current / width);

        area++;
        sumX += x;
        sumY += y;

        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;

        let isBoundary = false;
        for (const [dx, dy] of neighbors4) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            perimeter++;
            isBoundary = true;
            continue;
          }

          const nIndex = ny * width + nx;
          if (mask[nIndex] === 0) {
            perimeter++;
            isBoundary = true;
          }
        }

        if (isBoundary) {
          boundary.push(current);
        }

        for (const [dx, dy] of neighbors8) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

          const nIndex = ny * width + nx;
          if (mask[nIndex] === 1 && !visited[nIndex]) {
            visited[nIndex] = 1;
            queue.push(nIndex);
          }
        }
      }

      if (area >= minArea && boundary.length >= 8) {
        components.push({
          area,
          sumX,
          sumY,
          boundary,
          perimeter,
          bbox: { minX, minY, maxX, maxY },
        });
      }
    }

    return components;
  }

  private orderContourByAngle(points: Point[], center: Point): Point[] {
    return [...points].sort((a, b) => {
      const angleA = Math.atan2(a.y - center.y, a.x - center.x);
      const angleB = Math.atan2(b.y - center.y, b.x - center.x);
      if (angleA !== angleB) return angleA - angleB;

      const distA = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
      const distB = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (b.y - center.y);
      return distA - distB;
    });
  }

  private simplifyContour(points: Point[], epsilon: number): Point[] {
    if (points.length <= 3) {
      return points;
    }

    const recurse = (start: number, end: number, input: Point[], output: Point[]) => {
      let maxDistance = 0;
      let maxIndex = -1;

      for (let i = start + 1; i < end; i++) {
        const distance = this.distancePointToSegment(input[i], input[start], input[end]);
        if (distance > maxDistance) {
          maxDistance = distance;
          maxIndex = i;
        }
      }

      if (maxDistance > epsilon && maxIndex !== -1) {
        recurse(start, maxIndex, input, output);
        recurse(maxIndex, end, input, output);
      } else {
        output.push(input[start]);
      }
    };

    const output: Point[] = [];
    recurse(0, points.length - 1, points, output);
    output.push(points[points.length - 1]);

    return output;
  }

  private distancePointToSegment(point: Point, a: Point, b: Point): number {
    const vx = b.x - a.x;
    const vy = b.y - a.y;
    const wx = point.x - a.x;
    const wy = point.y - a.y;

    const c1 = vx * wx + vy * wy;
    if (c1 <= 0) return Math.hypot(point.x - a.x, point.y - a.y);

    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) return Math.hypot(point.x - b.x, point.y - b.y);

    const t = c1 / c2;
    const px = a.x + t * vx;
    const py = a.y + t * vy;
    return Math.hypot(point.x - px, point.y - py);
  }

  private removeNearCollinear(points: Point[]): Point[] {
    if (points.length <= 3) return points;

    const filtered: Point[] = [];
    const n = points.length;

    for (let i = 0; i < n; i++) {
      const prev = points[(i - 1 + n) % n];
      const current = points[i];
      const next = points[(i + 1) % n];

      const v1x = current.x - prev.x;
      const v1y = current.y - prev.y;
      const v2x = next.x - current.x;
      const v2y = next.y - current.y;

      const cross = Math.abs(v1x * v2y - v1y * v2x);
      const norm = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
      const normalizedCross = norm > 0 ? cross / norm : 0;

      if (normalizedCross > 0.06 || points.length <= 6) {
        filtered.push(current);
      }
    }

    return filtered.length >= 3 ? filtered : points;
  }

  private simplifyClosedPolygon(points: Point[], epsilon: number): Point[] {
    if (points.length <= 3) return points;

    const closed = [...points, points[0]];
    const simplifiedOpen = this.simplifyContour(closed, epsilon);
    const deduped = simplifiedOpen.slice(0, -1);
    const cleaned = this.removeNearCollinear(deduped);
    return cleaned.length >= 3 ? cleaned : deduped;
  }

  private computeClosedPerimeter(points: Point[]): number {
    if (points.length < 2) return 0;

    let perimeter = 0;
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      perimeter += Math.hypot(b.x - a.x, b.y - a.y);
    }
    return perimeter;
  }

  private computeConvexHull(points: Point[]): Point[] {
    if (points.length <= 2) return points;

    const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
    const cross = (o: Point, a: Point, b: Point) =>
      (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    const lower: Point[] = [];
    for (const p of sorted) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
        lower.pop();
      }
      lower.push(p);
    }

    const upper: Point[] = [];
    for (let i = sorted.length - 1; i >= 0; i--) {
      const p = sorted[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
        upper.pop();
      }
      upper.push(p);
    }

    lower.pop();
    upper.pop();
    return lower.concat(upper);
  }

  private computePolygonArea(points: Point[]): number {
    if (points.length < 3) return 0;

    let area = 0;
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      area += a.x * b.y - b.x * a.y;
    }

    return area / 2;
  }

  private calculateConcavityRatio(points: Point[]): number {
    if (points.length < 4) return 0;

    const orientation = Math.sign(this.computePolygonArea(points)) || 1;
    let concaveCount = 0;

    for (let i = 0; i < points.length; i++) {
      const prev = points[(i - 1 + points.length) % points.length];
      const current = points[i];
      const next = points[(i + 1) % points.length];

      const v1x = current.x - prev.x;
      const v1y = current.y - prev.y;
      const v2x = next.x - current.x;
      const v2y = next.y - current.y;
      const cross = v1x * v2y - v1y * v2x;

      if (Math.sign(cross) !== orientation && Math.abs(cross) > 1e-3) {
        concaveCount++;
      }
    }

    return concaveCount / points.length;
  }

  private computeRectangleAngleScore(points: Point[]): number {
    if (points.length !== 4) return 0;

    let score = 0;
    for (let i = 0; i < 4; i++) {
      const prev = points[(i - 1 + 4) % 4];
      const current = points[i];
      const next = points[(i + 1) % 4];

      const ax = prev.x - current.x;
      const ay = prev.y - current.y;
      const bx = next.x - current.x;
      const by = next.y - current.y;

      const normA = Math.hypot(ax, ay);
      const normB = Math.hypot(bx, by);
      if (normA === 0 || normB === 0) return 0;

      const cosTheta = Math.max(-1, Math.min(1, (ax * bx + ay * by) / (normA * normB)));
      const angle = (Math.acos(cosTheta) * 180) / Math.PI;
      const deviation = Math.abs(90 - angle);
      score += Math.max(0, 1 - deviation / 25);
    }

    return score / 4;
  }

  private computeRadialFeatures(
    points: Point[],
    center: Point
  ): { coefficientOfVariation: number; peakCount: number } {
    if (points.length < 12) {
      return { coefficientOfVariation: 0, peakCount: 0 };
    }

    const bins = 72;
    const radialByBin = new Float32Array(bins);
    const binCounts = new Uint16Array(bins);
    const twoPi = Math.PI * 2;

    for (const p of points) {
      const dx = p.x - center.x;
      const dy = p.y - center.y;
      const radius = Math.hypot(dx, dy);
      let angle = Math.atan2(dy, dx);
      if (angle < 0) angle += twoPi;

      const bin = Math.min(bins - 1, Math.floor((angle / twoPi) * bins));
      radialByBin[bin] += radius;
      binCounts[bin]++;
    }

    for (let i = 0; i < bins; i++) {
      if (binCounts[i] > 0) {
        radialByBin[i] /= binCounts[i];
      }
    }

    // Fill empty bins by nearest known values to keep the signature continuous.
    for (let i = 0; i < bins; i++) {
      if (binCounts[i] > 0) continue;

      let left = (i - 1 + bins) % bins;
      while (left !== i && binCounts[left] === 0) {
        left = (left - 1 + bins) % bins;
      }

      let right = (i + 1) % bins;
      while (right !== i && binCounts[right] === 0) {
        right = (right + 1) % bins;
      }

      if (binCounts[left] > 0 && binCounts[right] > 0) {
        radialByBin[i] = (radialByBin[left] + radialByBin[right]) / 2;
      } else if (binCounts[left] > 0) {
        radialByBin[i] = radialByBin[left];
      } else if (binCounts[right] > 0) {
        radialByBin[i] = radialByBin[right];
      }
    }

    const smoothed = new Float32Array(bins);
    for (let i = 0; i < bins; i++) {
      const prev = radialByBin[(i - 1 + bins) % bins];
      const curr = radialByBin[i];
      const next = radialByBin[(i + 1) % bins];
      smoothed[i] = (prev + 2 * curr + next) / 4;
    }

    let mean = 0;
    for (let i = 0; i < bins; i++) {
      mean += smoothed[i];
    }
    mean /= bins;

    if (mean <= 1e-6) {
      return { coefficientOfVariation: 0, peakCount: 0 };
    }

    let variance = 0;
    for (let i = 0; i < bins; i++) {
      const diff = smoothed[i] - mean;
      variance += diff * diff;
    }
    variance /= bins;

    let peakCount = 0;
    const peakThreshold = mean * 1.08;
    for (let i = 0; i < bins; i++) {
      const prev = smoothed[(i - 1 + bins) % bins];
      const curr = smoothed[i];
      const next = smoothed[(i + 1) % bins];
      if (curr > prev && curr > next && curr > peakThreshold) {
        peakCount++;
      }
    }

    return {
      coefficientOfVariation: Math.sqrt(variance) / mean,
      peakCount,
    };
  }

  private classifyShape(features: {
    vertexCount: number;
    convexVertexCount: number;
    circularity: number;
    aspectRatio: number;
    extent: number;
    solidity: number;
    concavityRatio: number;
    radialVariation: number;
    radialPeaks: number;
    compactness: number;
    area: number;
    rectangleAngleScore: number;
  }): { type: DetectedShape["type"]; confidence: number } {
    const {
      vertexCount,
      convexVertexCount,
      circularity,
      aspectRatio,
      extent,
      solidity,
      concavityRatio,
      radialVariation,
      radialPeaks,
      compactness,
      area,
      rectangleAngleScore,
    } = features;

    const aspectScore = Math.max(0, 1 - Math.abs(1 - aspectRatio));
    const vertexCircleScore = Math.min(1, Math.max(0, (vertexCount - 6) / 8));
    const starPeakScore = Math.max(0, 1 - Math.abs(radialPeaks - 5) / 4);
    const roundnessScore =
      0.45 * Math.max(0, 1 - Math.abs(circularity - 1) / 0.45) +
      0.25 * aspectScore +
      0.3 * Math.max(0, 1 - radialVariation / 0.16);
    const likelyRound =
      aspectScore > 0.9 &&
      concavityRatio < 0.08 &&
      radialVariation < 0.12 &&
      (convexVertexCount >= 6 || radialPeaks <= 2);

    const likelyRectangle =
      convexVertexCount === 4 &&
      rectangleAngleScore > 0.6 &&
      concavityRatio < 0.1 &&
      extent > 0.62;

    const starCandidate =
      area >= 500 &&
      vertexCount >= 8 &&
      convexVertexCount >= 5 &&
      concavityRatio > 0.14 &&
      radialVariation > 0.14 &&
      radialPeaks >= 4 &&
      solidity < 0.84;

    // Deterministic priors for clean synthetic shapes.
    if (likelyRectangle) {
      return { type: "rectangle", confidence: 0.92 };
    }
    if (likelyRound && (circularity > 0.74 || roundnessScore > 0.78)) {
      return { type: "circle", confidence: Math.min(0.99, 0.82 + (circularity - 0.84)) };
    }
    if (convexVertexCount === 3 && concavityRatio < 0.1) {
      return { type: "triangle", confidence: 0.9 };
    }
    if (
      area < 900 &&
      concavityRatio < 0.12 &&
      convexVertexCount <= 5 &&
      rectangleAngleScore < 0.5 &&
      circularity < 0.78
    ) {
      return { type: "triangle", confidence: 0.82 };
    }
    if (
      convexVertexCount === 5 &&
      concavityRatio < 0.1 &&
      solidity > 0.9 &&
      radialPeaks >= 3 &&
      radialVariation > 0.06
    ) {
      return { type: "pentagon", confidence: 0.88 };
    }
    if (convexVertexCount === 4 && rectangleAngleScore > 0.5 && concavityRatio < 0.1) {
      return { type: "rectangle", confidence: 0.9 };
    }
    if (starCandidate) {
      return { type: "star", confidence: 0.86 };
    }

    const scores: Record<DetectedShape["type"], number> = {
      triangle:
        0.55 * Math.max(0, 1 - Math.abs(convexVertexCount - 3) / 2) +
        0.2 * Math.max(0, 1 - concavityRatio * 5) +
        0.15 * Math.min(1, solidity) +
        0.1 * Math.max(0, 1 - Math.abs(extent - 0.5) / 0.35),
      rectangle:
        0.4 * Math.max(0, 1 - Math.abs(convexVertexCount - 4) / 2) +
        0.35 * rectangleAngleScore +
        0.15 * Math.max(0, 1 - concavityRatio * 5) +
        0.1 * Math.max(0, 1 - Math.abs(extent - 0.75) / 0.35),
      pentagon:
        0.5 * Math.max(0, 1 - Math.abs(convexVertexCount - 5) / 2) +
        0.25 * Math.min(1, solidity) +
        0.15 * Math.max(0, 1 - concavityRatio * 5) +
        0.1 * Math.max(0, 1 - Math.abs(extent - 0.68) / 0.3),
      star:
        0.25 * Math.max(0, 1 - Math.abs(vertexCount - 10) / 8) +
        0.25 * Math.min(1, concavityRatio * 3.2) +
        0.2 * Math.max(0, Math.min(1, (0.9 - solidity) / 0.38)) +
        0.2 * Math.min(1, radialVariation / 0.24) +
        0.1 * starPeakScore,
      circle:
        0.45 * Math.max(0, 1 - Math.abs(circularity - 1) / 0.35) +
        0.2 * aspectScore +
        0.2 * vertexCircleScore +
        0.15 * Math.max(0, 1 - radialVariation / 0.2),
    };

    // Extra guardrails for common confusions.
    if (convexVertexCount === 4 && rectangleAngleScore > 0.55) {
      scores.rectangle += 0.25;
    }
    if (likelyRound) {
      scores.circle += 0.28;
      scores.pentagon *= 0.62;
    }
    if (concavityRatio < 0.08) {
      scores.star *= 0.4;
    }
    if (area < 500 || vertexCount < 8 || convexVertexCount < 5) {
      scores.star *= 0.2;
    }
    if (convexVertexCount === 3 && concavityRatio < 0.12) {
      scores.triangle += 0.2;
      scores.star *= 0.2;
    }
    if (area < 1200) {
      scores.pentagon *= 0.45;
      scores.triangle += 0.1;
    }
    if (radialVariation > 0.18 && concavityRatio > 0.15) {
      scores.star += 0.2;
      scores.circle *= 0.55;
    }
    if (circularity > 0.84 && concavityRatio < 0.1 && radialVariation < 0.14) {
      scores.circle += 0.24;
    }
    if (compactness < 0.02 && extent < 0.2) {
      scores.rectangle *= 0.7;
      scores.pentagon *= 0.7;
    }

    let bestType: DetectedShape["type"] = "rectangle";
    let bestScore = -Infinity;
    for (const [type, score] of Object.entries(scores) as Array<
      [DetectedShape["type"], number]
    >) {
      if (score > bestScore) {
        bestScore = score;
        bestType = type;
      }
    }

    const confidence = Math.max(0.45, Math.min(0.99, bestScore));
    return { type: bestType, confidence };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});

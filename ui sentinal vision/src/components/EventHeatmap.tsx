import { useEffect, useRef } from "react";

interface EventHeatmapProps {
  events: { x: number; y: number; intensity: number }[];
}

const EventHeatmap = ({ events }: EventHeatmapProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;

    // Light base with grid
    ctx.fillStyle = "#f0f4f8";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(0, 0, 0, 0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 25) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 25) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    // Draw heat
    events.forEach(e => {
      const px = (e.x / 100) * w, py = (e.y / 100) * h;
      const r = 40 + e.intensity * 30;
      const g = ctx.createRadialGradient(px, py, 0, px, py, r);

      if (e.intensity > 0.7) {
        g.addColorStop(0, `rgba(239, 68, 68, ${0.5 * e.intensity})`);
        g.addColorStop(0.5, `rgba(251, 146, 60, ${0.25 * e.intensity})`);
      } else {
        g.addColorStop(0, `rgba(14, 165, 233, ${0.4 * e.intensity})`);
        g.addColorStop(0.5, `rgba(56, 189, 248, ${0.2 * e.intensity})`);
      }
      g.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = g;
      ctx.fillRect(px - r, py - r, r * 2, r * 2);
    });

    // Camera label
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.font = "11px 'Space Mono', monospace";
    ctx.fillText("cam1 Coverage Area", 8, 16);
  }, [events]);

  return (
    <canvas ref={canvasRef} width={600} height={350} className="w-full h-auto rounded-lg border border-border" />
  );
};

export default EventHeatmap;

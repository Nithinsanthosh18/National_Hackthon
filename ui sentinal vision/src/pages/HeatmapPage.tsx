import DashboardLayout from "@/components/DashboardLayout";
import EventHeatmap from "@/components/EventHeatmap";
import { useMonitoring } from "@/context/MonitoringContext";

const HeatmapPage = () => {
  return (
    <DashboardLayout title="Live Video Event Heatmap">
      <div className="max-w-[1200px] mx-auto space-y-5">
        <p className="text-muted-foreground text-sm">
          Visual representation of detection activity zones. The live video overlay accumulates detections over time. High density areas (hot spots) appear in red, while low density areas appear in blue.
        </p>
        <div className="bg-card border border-border rounded-lg p-5">
          <h3 className="text-sm font-semibold text-foreground mb-3">Camera 1 — Main Entrance</h3>
          <div className="relative rounded-lg overflow-hidden bg-black/20 aspect-video w-full flex items-center justify-center">
            {/* Direct feed from the backend which now includes the blended heatmap overlay */}
            <img
              src="/heatmap-feed/cam1"
              alt="Heatmap Feed Stream"
              className="w-full h-full object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
                (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
              }}
            />
            <div className="absolute hidden text-muted-foreground flex flex-col items-center">
              <span>Camera Feed Offline</span>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default HeatmapPage;

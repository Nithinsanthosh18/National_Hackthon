import DashboardLayout from "@/components/DashboardLayout";
import LiveStats from "@/components/LiveStats";
import CameraFeed from "@/components/CameraFeed";
import AlertsPanel from "@/components/AlertsPanel";
import EventHeatmap from "@/components/EventHeatmap";
import { useMonitoring } from "@/context/MonitoringContext";
import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

const Index = () => {
  const { alerts, heatPoints, totalDetections, detectionStats, activeCameras, fps, handleDetection } = useMonitoring();

  return (
    <DashboardLayout title="Dashboard">
      <div className="space-y-6 max-w-[1400px] mx-auto">
        <LiveStats totalDetections={totalDetections} detectionStats={detectionStats} activeCameras={activeCameras} alertCount={alerts.length} fps={fps} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Live Preview</h2>
              <Link to="/cameras" className="text-xs text-primary flex items-center gap-1 hover:underline">
                All cameras <ArrowRight className="h-3 w-3" />
              </Link>
            </div>
            <CameraFeed name="Main Entrance" camId="cam1" isActive compact onDetection={handleDetection} />
          </div>
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Recent Alerts</h2>
              <Link to="/alerts" className="text-xs text-primary flex items-center gap-1 hover:underline">
                View all <ArrowRight className="h-3 w-3" />
              </Link>
            </div>
            <AlertsPanel alerts={alerts} compact />
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Activity Heatmap</h2>
            <Link to="/heatmap" className="text-xs text-primary flex items-center gap-1 hover:underline">
              Full view <ArrowRight className="h-3 w-3" />
            </Link>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="relative rounded-lg overflow-hidden bg-black/20 aspect-video w-full flex items-center justify-center">
              <img
                src="/heatmap-feed/cam1"
                alt="Activity Heatmap Feed"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                  (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                }}
              />
              <div className="absolute hidden text-muted-foreground flex flex-col items-center">
                <span>Heatmap Feed Offline</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Index;

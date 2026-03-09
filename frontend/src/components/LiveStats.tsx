import { Activity, Users, Eye, Bell } from "lucide-react";

interface LiveStatsProps {
  totalDetections: number;
  detectionStats: Record<string, number>;
  activeCameras: number;
  alertCount: number;
  fps: number;
}

const LiveStats = ({ totalDetections, detectionStats, activeCameras, alertCount, fps }: LiveStatsProps) => {
  const weaponCount = detectionStats["weapon"] || 0;

  const stats = [
    { icon: Users, label: "Total Detections", value: totalDetections, iconClass: "text-info" },
    { icon: Eye, label: "Active Cameras", value: activeCameras, iconClass: "text-success" },
    { icon: Bell, label: "Weapons Detected", value: weaponCount, iconClass: "text-destructive" },
    { icon: Activity, label: "Avg FPS", value: fps, iconClass: "text-primary" },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((s) => (
        <div key={s.label} className="bg-card border border-border rounded-lg p-4 card-hover">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-muted flex items-center justify-center">
              <s.icon className={`h-5 w-5 ${s.iconClass}`} />
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground">{s.value}</p>
              <p className="text-xs text-muted-foreground">{s.label}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default LiveStats;

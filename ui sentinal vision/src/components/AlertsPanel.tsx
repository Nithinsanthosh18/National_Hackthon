import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface AlertEvent {
  id: string;
  type: string;
  severity: "HIGH" | "MEDIUM" | "LOW";
  timestamp: string;
  camera: string;
}

interface AlertsPanelProps {
  alerts: AlertEvent[];
  compact?: boolean;
}

const severityBadge: Record<string, string> = {
  HIGH: "bg-destructive text-destructive-foreground",
  MEDIUM: "bg-warning text-warning-foreground",
  LOW: "bg-accent text-accent-foreground",
};

const severityBorder: Record<string, string> = {
  HIGH: "border-l-destructive",
  MEDIUM: "border-l-warning",
  LOW: "border-l-primary",
};

const AlertsPanel = ({ alerts, compact = false }: AlertsPanelProps) => {
  const displayed = compact ? alerts.slice(0, 5) : alerts;
  return (
    <div className="space-y-2">
      {displayed.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">No alerts at this time</div>
      )}
      {displayed.map((alert) => (
        <div
          key={alert.id}
          className={`flex items-center justify-between px-4 py-3 bg-card rounded-lg border border-border border-l-4 ${severityBorder[alert.severity]} card-hover animate-slide-up`}
        >
          <div className="flex items-center gap-3">
            <AlertTriangle className={`h-4 w-4 ${alert.severity === "HIGH" ? "text-destructive" : alert.severity === "MEDIUM" ? "text-warning" : "text-primary"}`} />
            <div>
              <p className="text-sm font-medium text-foreground">{alert.type}</p>
              <p className="text-xs text-muted-foreground">{alert.camera} · {alert.timestamp}</p>
            </div>
          </div>
          <Badge className={severityBadge[alert.severity]}>{alert.severity}</Badge>
        </div>
      ))}
    </div>
  );
};

export default AlertsPanel;
export type { AlertEvent as AlertEventType };

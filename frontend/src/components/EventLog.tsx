import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";

export interface LogEntry {
  id: string;
  timestamp: string;
  camera: string;
  event: string;
  details: string;
  severity: "HIGH" | "MEDIUM" | "LOW";
}

interface EventLogProps {
  logs: LogEntry[];
}

const severityStyle: Record<string, string> = {
  HIGH: "bg-destructive/10 text-destructive border-destructive/20",
  MEDIUM: "bg-warning/10 text-warning border-warning/20",
  LOW: "bg-primary/10 text-primary border-primary/20",
};

const EventLog = ({ logs }: EventLogProps) => {
  return (
    <ScrollArea className="h-[400px]">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-card z-10">
          <tr className="border-b border-border text-muted-foreground text-xs uppercase tracking-wider">
            <th className="text-left py-3 px-4">Time</th>
            <th className="text-left py-3 px-4">Camera</th>
            <th className="text-left py-3 px-4">Event</th>
            <th className="text-left py-3 px-4">Details</th>
            <th className="text-left py-3 px-4">Severity</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log) => (
            <tr key={log.id} className="border-b border-border/50 hover:bg-muted/50 transition-colors">
              <td className="py-3 px-4 font-mono text-xs text-muted-foreground">{log.timestamp}</td>
              <td className="py-3 px-4 font-medium">{log.camera}</td>
              <td className="py-3 px-4">{log.event}</td>
              <td className="py-3 px-4 text-muted-foreground text-xs">{log.details}</td>
              <td className="py-3 px-4">
                <Badge variant="outline" className={severityStyle[log.severity]}>{log.severity}</Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollArea>
  );
};

export default EventLog;

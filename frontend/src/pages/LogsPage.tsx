import DashboardLayout from "@/components/DashboardLayout";
import EventLog from "@/components/EventLog";
import { useMonitoring } from "@/context/MonitoringContext";

const LogsPage = () => {
  const { logs } = useMonitoring();

  return (
    <DashboardLayout title="Incident Logs">
      <div className="max-w-[1200px] mx-auto">
        <p className="text-muted-foreground text-sm mb-5">
          Complete history of all detected events across all cameras.
        </p>
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <EventLog logs={logs} />
        </div>
      </div>
    </DashboardLayout>
  );
};

export default LogsPage;

import DashboardLayout from "@/components/DashboardLayout";
import AlertsPanel from "@/components/AlertsPanel";
import { useMonitoring } from "@/context/MonitoringContext";
import { Badge } from "@/components/ui/badge";

const AlertsPage = () => {
  const { alerts } = useMonitoring();

  const personAlerts = alerts.filter(a => a.type.includes("PERSON"));
  const weaponAlerts = alerts.filter(a => a.type.includes("WEAPON") || a.type.includes("KNIFE") || a.type.includes("GUN"));
  const objectAlerts = alerts.filter(a => !a.type.includes("PERSON") && !a.type.includes("WEAPON") && !a.type.includes("KNIFE") && !a.type.includes("GUN"));

  return (
    <DashboardLayout title="Real-Time Alerts">
      <div className="max-w-[1000px] mx-auto space-y-8">

        {/* Critical Weapons Section */}
        {weaponAlerts.length > 0 && (
          <div className="bg-destructive/5 border border-destructive/20 p-4 rounded-xl">
            <h2 className="text-lg font-bold text-destructive border-b border-destructive/20 pb-2 mb-4 flex items-center gap-2">
              <Badge variant="destructive" className="px-2">CRITICAL</Badge> Weapon Detections
            </h2>
            <AlertsPanel alerts={weaponAlerts} />
          </div>
        )}

        {/* Person Detections Section */}
        <div>
          <h2 className="text-lg font-semibold text-foreground border-b border-border pb-2 mb-4">Person Detections</h2>
          <AlertsPanel alerts={personAlerts} />
        </div>

        {/* Other Object Detections Section */}
        <div>
          <h2 className="text-lg font-semibold text-foreground border-b border-border pb-2 mb-4">Other Objects (Vehicles, Baggage, etc.)</h2>
          <AlertsPanel alerts={objectAlerts} />
        </div>

      </div>
    </DashboardLayout>
  );
};

export default AlertsPage;

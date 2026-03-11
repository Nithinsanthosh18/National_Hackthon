import DashboardLayout from "@/components/DashboardLayout";
import { useMonitoring } from "@/context/MonitoringContext";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend } from "recharts";
import { Download, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";

const categoryColors: Record<string, string> = {
  "Person": "#0ea5e9",
  "Vehicle": "#f59e0b",
  "Weapon": "#ef4444",
  "Alerts": "#ef4444",
  "Unknown": "#94a3b8",
};

const ReportsPage = () => {
  const { alerts, logs, totalDetections, detectionStats, activeCameras, fps } = useMonitoring();
  const [hourlyData, setHourlyData] = useState<any[]>([]);
  const [weeklyData, setWeeklyData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const [hourlyRes, weeklyRes] = await Promise.all([
          fetch("/stats/hourly"),
          fetch("/stats/weekly")
        ]);

        if (hourlyRes.ok) setHourlyData(await hourlyRes.json());
        if (weeklyRes.ok) setWeeklyData(await weeklyRes.json());
      } catch (err) {
        console.error("Failed to fetch analytics", err);
      } finally {
        setLoading(false);
      }
    };
    fetchAnalytics();
  }, []);

  const exportCSV = () => {
    const headers = "Timestamp,Camera,Event,Details,Severity\n";
    const rows = logs.map(l => `${l.timestamp},${l.camera},${l.event},${l.details},${l.severity}`).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "sentinel_report.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  const generatePDF = () => {
    const doc = new jsPDF();

    // Header
    doc.setFontSize(22);
    doc.setTextColor(14, 165, 233); // Primary color
    doc.text("SENTINEL VISION", 105, 20, { align: "center" });
    doc.setFontSize(16);
    doc.setTextColor(100);
    doc.text("Security Intelligence Report", 105, 30, { align: "center" });

    doc.setDrawColor(200);
    doc.line(20, 35, 190, 35);

    // Summary
    doc.setFontSize(12);
    doc.setTextColor(0);
    doc.text(`Report Generated: ${new Date().toLocaleString()}`, 20, 45);
    doc.text(`Total Detections Today: ${totalDetections}`, 20, 52);
    doc.text(`Active Alerts: ${alerts.length}`, 20, 59);
    doc.text(`Operational Cameras: ${activeCameras}`, 20, 66);

    // Table of Recent Events
    doc.setFontSize(14);
    doc.text("Recent Security Events", 20, 80);

    const tableData = logs.slice(0, 15).map(l => [
      l.timestamp,
      l.event,
      l.details,
      l.severity
    ]);

    autoTable(doc, {
      startY: 85,
      head: [['Timestamp', 'Event', 'Details', 'Severity']],
      body: tableData,
      headStyles: { fillColor: [14, 165, 233] },
    });

    // Footer
    const pageCount = doc.getNumberOfPages();
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(10);
      doc.setTextColor(150);
      doc.text(`Page ${i} of ${pageCount}`, 105, 285, { align: "center" });
      doc.text("Proprietary & Confidential - Sentinel Vision", 105, 290, { align: "center" });
    }

    doc.save("Sentinel_Vision_Report.pdf");
  };

  // Prepare category data for the pie chart
  const categoryData = Object.entries(detectionStats).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value,
    color: categoryColors[name.charAt(0).toUpperCase() + name.slice(1)] || categoryColors["Unknown"]
  }));

  return (
    <DashboardLayout title="Reports & Analytics">
      <div className="max-w-[1400px] mx-auto space-y-6">
        {/* Summary Cards */}
        <div className="flex items-center justify-between">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 flex-1">
            {[
              { label: "Total Detections", value: totalDetections },
              { label: "Alerts Triggered", value: alerts.length },
              { label: "Cameras Active", value: activeCameras },
              { label: "Stream Quality", value: `${fps.toFixed(1)} FPS` },
            ].map(s => (
              <div key={s.label} className="bg-card border border-border rounded-lg p-4">
                <p className="text-2xl font-bold text-foreground">{s.value}</p>
                <p className="text-xs text-muted-foreground">{s.label}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="gap-1.5" onClick={exportCSV}>
            <Download className="h-4 w-4" /> Export CSV
          </Button>
          <Button size="sm" variant="outline" className="gap-1.5" onClick={generatePDF}>
            <FileText className="h-4 w-4" /> Generate PDF Report
          </Button>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Hourly Activity */}
          <div className="bg-card border border-border rounded-lg p-5">
            <h3 className="text-sm font-semibold text-foreground mb-4">Hourly Detection Activity (24h)</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} stroke="hsl(215, 10%, 50%)" />
                <YAxis tick={{ fontSize: 11 }} stroke="hsl(215, 10%, 50%)" />
                <Tooltip contentStyle={{ background: "hsl(0, 0%, 100%)", border: "1px solid hsl(210, 15%, 90%)", borderRadius: 8 }} />
                <Bar dataKey="detections" fill="hsl(172, 66%, 40%)" radius={[4, 4, 0, 0]} name="Detections" />
                <Bar dataKey="alerts" fill="hsl(0, 72%, 55%)" radius={[4, 4, 0, 0]} name="Alerts" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detection Categories Breakdown */}
          <div className="bg-card border border-border rounded-lg p-5">
            <h3 className="text-sm font-semibold text-foreground mb-4">Detection Categories Breakdown</h3>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={categoryData.length > 0 ? categoryData : [{ name: "No Data", value: 1, color: "#94a3b8" }]} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={4} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {categoryData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                  {categoryData.length === 0 && <Cell fill="#94a3b8" />}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Weekly Trend Analysis */}
          <div className="bg-card border border-border rounded-lg p-5 lg:col-span-2">
            <h3 className="text-sm font-semibold text-foreground mb-4">Weekly Trend Analysis</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 15%, 90%)" />
                <XAxis dataKey="day" stroke="hsl(215, 10%, 50%)" />
                <YAxis stroke="hsl(215, 10%, 50%)" />
                <Tooltip contentStyle={{ background: "hsl(0, 0%, 100%)", border: "1px solid hsl(210, 15%, 90%)", borderRadius: 8 }} />
                <Legend />
                <Line type="monotone" dataKey="person" stroke="#0ea5e9" strokeWidth={2} dot={{ r: 4 }} name="Persons" />
                <Line type="monotone" dataKey="vehicle" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4 }} name="Vehicles" />
                <Line type="monotone" dataKey="alert" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} name="Alerts" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default ReportsPage;

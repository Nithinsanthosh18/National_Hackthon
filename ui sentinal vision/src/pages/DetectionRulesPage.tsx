import DashboardLayout from "@/components/DashboardLayout";
import { useState, useEffect } from "react";
import { Plus, Trash2, Edit, Shield, AlertTriangle, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

interface DetectionRule {
  id: string;
  name: string;
  category: "Object" | "Behavior" | "Zone" | "Facial";
  target: string;
  cameras: string[];
  confidenceThreshold: number;
  alertSeverity: "High" | "Medium" | "Low";
  enabled: boolean;
  description: string;
}

const initialRules: DetectionRule[] = [
  { id: "1", name: "Weapon Detection", category: "Object", target: "Firearms, Knives, Explosives", cameras: ["cam1"], confidenceThreshold: 0.6, alertSeverity: "High", enabled: true, description: "Detect visible weapons including firearms, knives, and explosive devices" },
  { id: "2", name: "Intrusion Alert", category: "Behavior", target: "Unauthorized entry", cameras: ["cam1"], confidenceThreshold: 0.7, alertSeverity: "High", enabled: true, description: "Alert when person detected in restricted zone outside permitted hours" },
  { id: "3", name: "Loitering Detection", category: "Behavior", target: "Suspicious lingering", cameras: ["cam1"], confidenceThreshold: 0.5, alertSeverity: "Medium", enabled: true, description: "Detect individuals remaining in area for extended period" },
  { id: "4", name: "Perimeter Breach", category: "Zone", target: "Fence line crossing", cameras: ["cam1"], confidenceThreshold: 0.75, alertSeverity: "High", enabled: true, description: "Detect movement crossing defined perimeter boundaries" },
  { id: "5", name: "Vehicle Tracking", category: "Object", target: "Unauthorized vehicles", cameras: ["cam1"], confidenceThreshold: 0.65, alertSeverity: "Medium", enabled: false, description: "Track and identify vehicles not in authorized database" },
  { id: "6", name: "Face Match", category: "Facial", target: "Watchlist individuals", cameras: ["cam1"], confidenceThreshold: 0.8, alertSeverity: "High", enabled: true, description: "Match detected faces against watchlist database" },
  { id: "7", name: "Crowd Density", category: "Behavior", target: "Unusual gathering", cameras: ["cam1"], confidenceThreshold: 0.6, alertSeverity: "Medium", enabled: true, description: "Alert when crowd density exceeds normal threshold" },
  { id: "8", name: "Abandoned Object", category: "Object", target: "Unattended bags/packages", cameras: ["cam1"], confidenceThreshold: 0.55, alertSeverity: "High", enabled: true, description: "Detect objects left unattended for configurable duration" },
];

const categoryColors: Record<string, string> = {
  Object: "bg-info/10 text-info border-info/20",
  Behavior: "bg-warning/10 text-warning border-warning/20",
  Zone: "bg-primary/10 text-primary border-primary/20",
  Facial: "bg-destructive/10 text-destructive border-destructive/20",
};

const severityColors: Record<string, string> = {
  High: "bg-destructive text-destructive-foreground",
  Medium: "bg-warning text-warning-foreground",
  Low: "bg-accent text-accent-foreground",
};

const DetectionRulesPage = () => {
  const [rules, setRules] = useState<DetectionRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newRule, setNewRule] = useState({
    name: "", category: "Object" as DetectionRule["category"], target: "",
    confidenceThreshold: 0.7, alertSeverity: "High" as DetectionRule["alertSeverity"], description: "",
  });

  const fetchRules = async () => {
    try {
      const response = await fetch("/rules");
      if (response.ok) {
        const data = await response.json();
        // Convert cameras string from backend back to array for frontend
        const formattedRules = data.rules.map((r: any) => ({
          ...r,
          cameras: r.cameras ? r.cameras.split(',') : []
        }));
        setRules(formattedRules);
      }
    } catch (err) {
      console.error("Failed to fetch rules", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRules();
  }, []);

  const toggleRule = async (id: string, currentStatus: boolean) => {
    try {
      const response = await fetch(`/rules/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !currentStatus ? 1 : 0 })
      });
      if (response.ok) {
        setRules(prev => prev.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r));
      }
    } catch (err) {
      console.error("Failed to toggle rule", err);
    }
  };

  const deleteRule = async (id: string) => {
    try {
      const response = await fetch(`/rules/${id}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        setRules(prev => prev.filter(r => r.id !== id));
      }
    } catch (err) {
      console.error("Failed to delete rule", err);
    }
  };

  const addRule = async () => {
    if (!newRule.name || !newRule.target) return;

    const payload = {
      ...newRule,
      cameras: "All" // Use "All" as default to ensure it applies broadly
    };

    try {
      const response = await fetch("/rules", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        fetchRules();
        setNewRule({ name: "", category: "Object", target: "", confidenceThreshold: 0.7, alertSeverity: "High", description: "" });
        setDialogOpen(false);
      }
    } catch (err) {
      console.error("Failed to add rule", err);
    }
  };

  const activeCount = rules.filter(r => r.enabled).length;

  return (
    <DashboardLayout title="Detection Rules">
      <div className="max-w-[1200px] mx-auto space-y-6">
        {/* Summary */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">
              Configure what the AI should detect — objects, behaviors, zones, and faces. {activeCount}/{rules.length} rules active.
            </p>
          </div>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="gap-1.5"><Plus className="h-4 w-4" /> New Rule</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>Create Detection Rule</DialogTitle></DialogHeader>
              <div className="space-y-4 mt-2">
                <div>
                  <Label>Rule Name</Label>
                  <Input placeholder="e.g., Tailgating Detection" value={newRule.name} onChange={e => setNewRule(p => ({ ...p, name: e.target.value }))} />
                </div>
                <div>
                  <Label>Category</Label>
                  <Select value={newRule.category} onValueChange={(v) => setNewRule(p => ({ ...p, category: v as DetectionRule["category"] }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Object">Object Detection</SelectItem>
                      <SelectItem value="Behavior">Behavior Analysis</SelectItem>
                      <SelectItem value="Zone">Zone Monitoring</SelectItem>
                      <SelectItem value="Facial">Facial Recognition</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Detection Target</Label>
                  <Input placeholder="What to look for (e.g. person, weapon)..." value={newRule.target} onChange={e => setNewRule(p => ({ ...p, target: e.target.value }))} />
                </div>
                <div>
                  <Label>Confidence Threshold: {Math.round(newRule.confidenceThreshold * 100)}%</Label>
                  <Slider value={[newRule.confidenceThreshold]} onValueChange={([v]) => setNewRule(p => ({ ...p, confidenceThreshold: v }))} min={0.1} max={1} step={0.05} className="mt-2" />
                </div>
                <div>
                  <Label>Alert Severity</Label>
                  <Select value={newRule.alertSeverity} onValueChange={(v) => setNewRule(p => ({ ...p, alertSeverity: v as DetectionRule["alertSeverity"] }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="High">High</SelectItem>
                      <SelectItem value="Medium">Medium</SelectItem>
                      <SelectItem value="Low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Description</Label>
                  <Input placeholder="Brief description..." value={newRule.description} onChange={e => setNewRule(p => ({ ...p, description: e.target.value }))} />
                </div>
                <Button onClick={addRule} className="w-full">Create Rule</Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Rules List */}
        <div className="space-y-3">
          {loading ? (
            <div className="text-center py-20 text-muted-foreground">Loading detection rules...</div>
          ) : rules.length === 0 ? (
            <div className="text-center py-20 border border-dashed border-border rounded-lg text-muted-foreground">
              No detection rules configured. Create one to start monitoring.
            </div>
          ) : (
            rules.map((rule) => (
              <div key={rule.id} className={`bg-card border border-border rounded-lg p-4 card-hover transition-opacity ${!rule.enabled ? "opacity-50" : ""}`}>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold text-foreground">{rule.name}</h3>
                      <Badge variant="outline" className={categoryColors[rule.category]}>{rule.category}</Badge>
                      <Badge className={severityColors[rule.alertSeverity]}>{rule.alertSeverity}</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{rule.description}</p>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>🎯 {rule.target}</span>
                      <span>📷 {rule.cameras}</span>
                      <span>Threshold: {Math.round(rule.confidenceThreshold * 100)}%</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Switch checked={!!rule.enabled} onCheckedChange={() => toggleRule(rule.id, !!rule.enabled)} />
                    <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => deleteRule(rule.id)}>
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </DashboardLayout>
  );
};

export default DetectionRulesPage;

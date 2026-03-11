import DashboardLayout from "@/components/DashboardLayout";
import { useState, useEffect } from "react";
import { Plus, Trash2, Search, UserCheck, UserX, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

interface WatchlistEntry {
  id: string;
  name: string;
  alias: string;
  threat: "Critical" | "High" | "Medium" | "Low";
  status: "Active" | "Apprehended" | "Under Surveillance";
  lastCamera?: string;
  matchCount?: number;
  addedDate?: string;
  notes?: string;
  image_url?: string;
}

const initialWatchlist: WatchlistEntry[] = [];

const threatColors: Record<string, string> = {
  Critical: "bg-destructive text-destructive-foreground",
  High: "bg-warning text-warning-foreground",
  Medium: "bg-info text-info-foreground",
  Low: "bg-accent text-accent-foreground",
};

const statusColors: Record<string, string> = {
  Active: "border-destructive/30 text-destructive",
  Apprehended: "border-success/30 text-success",
  "Under Surveillance": "border-warning/30 text-warning",
};

const WatchlistPage = () => {
  const [entries, setEntries] = useState<WatchlistEntry[]>(initialWatchlist);
  const [searchQuery, setSearchQuery] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newEntry, setNewEntry] = useState({
    name: "", alias: "", threat: "Medium" as WatchlistEntry["threat"], notes: "", file: null as File | null
  });

  const fetchWatchlist = async () => {
    try {
      const res = await fetch("/watchlist");
      const data = await res.json();
      setEntries(data.suspects.map((s: any) => ({
        id: s.id,
        name: s.name,
        alias: s.alias,
        threat: s.threat as WatchlistEntry["threat"],
        status: s.status as WatchlistEntry["status"],
        notes: s.notes,
        addedDate: s.added_date,
        image_url: s.image_url
      })));
    } catch (err) {
      console.error("Failed to fetch watchlist", err);
    }
  };

  useEffect(() => { fetchWatchlist(); }, []);

  const filtered = entries.filter(e =>
    e.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    e.alias.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const addEntry = async () => {
    if (!newEntry.name || !newEntry.file) {
      alert("Name and a face photo are required!");
      return;
    }

    const formData = new FormData();
    formData.append("name", newEntry.name);
    formData.append("alias", newEntry.alias);
    formData.append("threat", newEntry.threat);
    formData.append("notes", newEntry.notes);
    formData.append("file", newEntry.file);

    try {
      const res = await fetch("/watchlist", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        fetchWatchlist();
        setNewEntry({ name: "", alias: "", threat: "Medium", notes: "", file: null });
        setDialogOpen(false);
      } else {
        const errorText = await res.text();
        alert(`Error adding suspect: ${errorText}`);
      }
    } catch (err) {
      console.error(err);
      alert("Network error.");
    }
  };

  const removeEntry = async (id: string) => {
    try {
      await fetch(`/watchlist/${id}`, { method: "DELETE" });
      setEntries(prev => prev.filter(e => e.id !== id));
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <DashboardLayout title="Watchlist & Facial Recognition">
      <div className="max-w-[1200px] mx-auto space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: "Total Subjects", value: entries.length, icon: "👤" },
            { label: "Active Threats", value: entries.filter(e => e.status === "Active").length, icon: "🔴" },
            { label: "Under Watch", value: entries.filter(e => e.status === "Under Surveillance").length, icon: "👁️" },
            { label: "Total Matches", value: entries.reduce((a, e) => a + e.matchCount, 0), icon: "🎯" },
          ].map(s => (
            <div key={s.label} className="bg-card border border-border rounded-lg p-4 card-hover">
              <div className="flex items-center gap-2">
                <span className="text-lg">{s.icon}</span>
                <div>
                  <p className="text-xl font-bold text-foreground">{s.value}</p>
                  <p className="text-xs text-muted-foreground">{s.label}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Search & Add */}
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search by name or alias..." className="pl-9" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
          </div>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="gap-1.5"><Plus className="h-4 w-4" /> Add Subject</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>Add to Watchlist</DialogTitle></DialogHeader>
              <div className="space-y-4 mt-2">
                <div><Label>Subject Name / ID</Label><Input placeholder="Subject identifier" value={newEntry.name} onChange={e => setNewEntry(p => ({ ...p, name: e.target.value }))} /></div>
                <div><Label>Known Alias</Label><Input placeholder="Alias (if known)" value={newEntry.alias} onChange={e => setNewEntry(p => ({ ...p, alias: e.target.value }))} /></div>
                <div>
                  <Label>Threat Level</Label>
                  <Select value={newEntry.threat} onValueChange={(v) => setNewEntry(p => ({ ...p, threat: v as WatchlistEntry["threat"] }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Critical">Critical</SelectItem>
                      <SelectItem value="High">High</SelectItem>
                      <SelectItem value="Medium">Medium</SelectItem>
                      <SelectItem value="Low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Face Photo</Label>
                  <Input type="file" accept="image/*" onChange={e => setNewEntry(p => ({ ...p, file: e.target.files ? e.target.files[0] : null }))} />
                </div>
                <Button onClick={addEntry} className="w-full">Upload & Add to Watchlist</Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Watchlist Entries */}
        <div className="space-y-3">
          {filtered.map((entry) => (
            <div key={entry.id} className="bg-card border border-border rounded-lg p-4 card-hover">
              <div className="flex items-start justify-between gap-4">
                <div className="flex gap-4 flex-1">
                  <div className="h-16 w-16 rounded-lg bg-muted flex items-center justify-center text-2xl shrink-0 overflow-hidden">
                    {entry.image_url ? (
                      <img src={entry.image_url} alt={entry.name} className="w-full h-full object-cover" />
                    ) : (
                      entry.status === "Apprehended" ? "✅" : "👤"
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <h3 className="font-semibold text-foreground">{entry.name}</h3>
                      {entry.alias !== "None" && entry.alias !== "Unknown" && (
                        <span className="text-xs text-muted-foreground italic">a.k.a. "{entry.alias}"</span>
                      )}
                      <Badge className={threatColors[entry.threat]}>{entry.threat}</Badge>
                      <Badge variant="outline" className={statusColors[entry.status]}>{entry.status}</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">{entry.notes}</p>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
                      <span>Camera: {entry.lastCamera || "None"}</span>
                      <span>Matches: {entry.matchCount || 0}</span>
                      <span>Added: {entry.addedDate || new Date().toISOString().split("T")[0]}</span>
                    </div>
                  </div>
                </div>
                <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" onClick={() => removeEntry(entry.id)}>
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </div>
            </div>
          ))}
          {filtered.length === 0 && (
            <p className="text-center text-muted-foreground py-8">No matching subjects found.</p>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
};

export default WatchlistPage;

import DashboardLayout from "@/components/DashboardLayout";
import CameraFeed from "@/components/CameraFeed";
import { useMonitoring } from "@/context/MonitoringContext";
import { useState, useEffect } from "react";
import { Plus, Trash2, Camera, Wifi, WifiOff, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface CameraSource {
  id: string;
  name: string;
  camId: string;
  type: "RTSP" | "Webcam" | "Drone" | "Body Cam" | "IP Camera";
  url: string;
  active: boolean;
  format: string;
}

const initialCameras: CameraSource[] = [];

const typeIcons: Record<string, string> = {
  "IP Camera": "📷",
  Webcam: "💻",
  Drone: "🛩️",
  "Body Cam": "🎥",
  RTSP: "📡",
};

const CamerasPage = () => {
  const { handleDetection } = useMonitoring();
  const [cameras, setCameras] = useState<CameraSource[]>(initialCameras);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newCam, setNewCam] = useState({ name: "", type: "IP Camera" as CameraSource["type"], url: "", format: "H.264" });

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const response = await fetch("/cameras");
        if (response.ok) {
          const data = await response.json();
          const loadedCameras = data.cameras.map((cam: any) => ({
            id: cam.id,
            name: cam.name,
            camId: cam.camId,
            type: cam.type,
            url: cam.url,
            active: cam.active,
            format: cam.format,
          }));
          setCameras(loadedCameras);
        }
      } catch (error) {
        console.error("Failed to fetch cameras:", error);
      }
    };

    fetchCameras();

    // Optional polling for status updates
    const interval = setInterval(fetchCameras, 10000);
    return () => clearInterval(interval);
  }, []);

  const addCamera = async () => {
    if (!newCam.name || !newCam.url) return;

    const id = String(Date.now());
    // Use lowercase cam ids to stay consistent with backend formats
    const camId = `cam_${Math.floor(Math.random() * 1000)}`;

    try {
      const response = await fetch("/cameras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newCam.name,
          camera_id: camId,
          url: newCam.url
        })
      });

      if (!response.ok) {
        console.error("Failed to add camera to background processor");
        return;
      }

      setCameras(prev => [...prev, {
        id,
        name: newCam.name,
        camId: camId,
        type: newCam.type,
        url: newCam.url,
        active: true,
        format: newCam.format,
      }]);
      setNewCam({ name: "", type: "IP Camera", url: "", format: "H.264" });
      setDialogOpen(false);

    } catch (err) {
      console.error("Error connecting source:", err);
    }
  };

  const removeCamera = async (id: string) => {
    const camToRemove = cameras.find(c => c.id === id);
    if (!camToRemove) return;

    try {
      const response = await fetch(`/cameras/${camToRemove.camId}`, {
        method: "DELETE"
      });

      if (response.ok) {
        setCameras(prev => prev.filter(c => c.id !== id));
      } else {
        console.error("Failed to delete camera from backend");
      }
    } catch (err) {
      console.error("Error deleting camera:", err);
    }
  };
  const toggleCamera = (id: string) => setCameras(prev => prev.map(c => c.id === id ? { ...c, active: !c.active } : c));

  return (
    <DashboardLayout title="Camera Feeds & Source Management">
      <div className="max-w-[1400px] mx-auto space-y-6">
        {/* Source Management Bar */}
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider">Connected Sources</h2>
              <p className="text-xs text-muted-foreground mt-1">
                {cameras.filter(c => c.active).length} active · {cameras.length} total · Supports RTSP, RTMP, UDP, HTTP, Webcam, MJPEG
              </p>
            </div>
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <DialogTrigger asChild>
                <Button size="sm" className="gap-1.5">
                  <Plus className="h-4 w-4" /> Add Source
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Video Source</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 mt-2">
                  <div>
                    <Label>Source Name</Label>
                    <Input placeholder="e.g., Rooftop Camera" value={newCam.name} onChange={e => setNewCam(p => ({ ...p, name: e.target.value }))} />
                  </div>
                  <div>
                    <Label>Source Type</Label>
                    <Select value={newCam.type} onValueChange={(v) => setNewCam(p => ({ ...p, type: v as CameraSource["type"] }))}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="IP Camera">IP Camera</SelectItem>
                        <SelectItem value="Webcam">Webcam</SelectItem>
                        <SelectItem value="Drone">Drone Feed</SelectItem>
                        <SelectItem value="Body Cam">Body Cam</SelectItem>
                        <SelectItem value="RTSP">RTSP Stream</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Stream URL / Source</Label>
                    <Input placeholder="rtsp://192.168.x.x:554/stream" value={newCam.url} onChange={e => setNewCam(p => ({ ...p, url: e.target.value }))} />
                  </div>
                  <div>
                    <Label>Format</Label>
                    <Select value={newCam.format} onValueChange={(v) => setNewCam(p => ({ ...p, format: v }))}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="H.264">H.264</SelectItem>
                        <SelectItem value="H.265">H.265 (HEVC)</SelectItem>
                        <SelectItem value="MJPEG">MJPEG</SelectItem>
                        <SelectItem value="VP8">VP8</SelectItem>
                        <SelectItem value="VP9">VP9</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button onClick={addCamera} className="w-full">Connect Source</Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          {/* Source List */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {cameras.map((cam) => (
              <div key={cam.id} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border border-border">
                <div className="flex items-center gap-3 min-w-0">
                  <span className="text-lg">{typeIcons[cam.type] || "📷"}</span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{cam.name}</p>
                    <p className="text-xs text-muted-foreground font-mono truncate">{cam.camId} · {cam.format}</p>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <Badge variant="outline" className={cam.active ? "border-success/30 text-success" : "border-muted-foreground/30 text-muted-foreground"}>
                    {cam.active ? "Online" : "Offline"}
                  </Badge>
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => toggleCamera(cam.id)}>
                    {cam.active ? <Wifi className="h-3.5 w-3.5 text-success" /> : <WifiOff className="h-3.5 w-3.5 text-muted-foreground" />}
                  </Button>
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => removeCamera(cam.id)}>
                    <Trash2 className="h-3.5 w-3.5 text-destructive" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Live Feeds Grid */}
        <div>
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-3">Live Feeds</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {cameras.filter(c => c.active).map((cam) => (
              <CameraFeed key={cam.id} name={cam.name} camId={cam.camId} isActive onDetection={handleDetection} />
            ))}
          </div>
          {cameras.filter(c => !c.active).length > 0 && (
            <>
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mt-6 mb-3">Offline Sources</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                {cameras.filter(c => !c.active).map((cam) => (
                  <CameraFeed key={cam.id} name={cam.name} camId={cam.camId} isActive={false} />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
};

export default CamerasPage;

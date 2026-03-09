import DashboardLayout from "@/components/DashboardLayout";
import { useState, useRef } from "react";
import { Upload, Play, Pause, FileVideo, Clock, CheckCircle2, AlertTriangle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ArchivePlayer } from "@/components/ArchivePlayer";
import { useMonitoring, ArchiveFile } from "@/context/MonitoringContext";

const statusIcons: Record<string, React.ReactNode> = {
  Queued: <Clock className="h-4 w-4 text-muted-foreground" />,
  Processing: <Loader2 className="h-4 w-4 text-info animate-spin" />,
  Completed: <CheckCircle2 className="h-4 w-4 text-success" />,
  Failed: <AlertTriangle className="h-4 w-4 text-destructive" />,
};

const statusBadge: Record<string, string> = {
  Queued: "bg-muted text-muted-foreground",
  Processing: "bg-info/10 text-info border-info/20",
  Completed: "bg-success/10 text-success border-success/20",
  Failed: "bg-destructive/10 text-destructive border-destructive/20",
};

const ArchivePage = () => {
  const { archiveFiles: files, addArchiveFiles, processArchiveFile } = useMonitoring();
  const [dragging, setDragging] = useState(false);
  const [activeVideo, setActiveVideo] = useState<string | null>(null);
  const [activeFindings, setActiveFindings] = useState<any[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFiles = Array.from(e.target.files);
      const newEntries: ArchiveFile[] = selectedFiles.map(file => ({
        id: String(Date.now() + Math.random()),
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(1) + " MB",
        duration: "Unknown",
        uploadDate: new Date().toISOString().split("T")[0],
        status: "Queued", progress: 0, findings: 0, source: "Manual Upload",
        fileObj: file
      }));
      addArchiveFiles(newEntries);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      const newEntries: ArchiveFile[] = droppedFiles.map(file => ({
        id: String(Date.now() + Math.random()),
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(1) + " MB",
        duration: "Unknown",
        uploadDate: new Date().toISOString().split("T")[0],
        status: "Queued", progress: 0, findings: 0, source: "Manual Upload",
        fileObj: file
      }));
      addArchiveFiles(newEntries);
    }
  };

  const startProcessing = (id: string) => {
    processArchiveFile(id);
  };

  const watchReport = (file: ArchiveFile) => {
    if (file.videoUrl) {
      setActiveVideo(file.videoUrl);
      setActiveFindings(file.findingsArray || []);
    }
  };

  return (
    <DashboardLayout title="Archive Analysis">
      <div className="max-w-[1200px] mx-auto space-y-6">
        <p className="text-sm text-muted-foreground">
          Upload recorded footage from any source — body cams, drones, surveillance cameras — for post-event AI analysis. Supports MP4, AVI, MKV, MOV, WEBM, FLV.
        </p>

        {activeVideo && (
          <div className="mb-8">
            <ArchivePlayer videoUrl={activeVideo} findings={activeFindings} onClose={() => { setActiveVideo(null); setActiveFindings([]); }} />
          </div>
        )}

        {/* Upload Zone */}
        <div
          className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${dragging ? "border-primary bg-accent/30" : "border-border bg-card"}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <Upload className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
          <h3 className="text-foreground font-semibold mb-1">Drop video files here</h3>
          <p className="text-sm text-muted-foreground mb-3">or click to browse — supports any video format</p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            className="hidden"
            multiple
            accept="video/*"
          />
          <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
            <FileVideo className="h-4 w-4 mr-1.5" /> Browse Files
          </Button>
        </div>

        {/* Analysis Queue */}
        <div>
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-3">Analysis Queue</h2>
          <div className="space-y-3">
            {files.map((file) => (
              <div key={file.id} className="bg-card border border-border rounded-lg p-4 card-hover">
                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    {statusIcons[file.status]}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <p className="text-sm font-medium text-foreground font-mono truncate">{file.name}</p>
                        <Badge variant="outline" className={statusBadge[file.status]}>{file.status}</Badge>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <span>{file.size}</span>
                        <span>{file.duration}</span>
                        <span>Source: {file.source}</span>
                        <span>Uploaded: {file.uploadDate}</span>
                        {file.status === "Completed" && <span className="text-success font-medium">{file.findings} findings</span>}
                      </div>
                      {file.status === "Processing" && (
                        <Progress value={file.progress} className="mt-2 h-1.5" />
                      )}
                    </div>
                  </div>
                  <div className="shrink-0">
                    {file.status === "Queued" && (
                      <Button size="sm" variant="outline" className="gap-1" onClick={() => startProcessing(file.id)}>
                        <Play className="h-3.5 w-3.5" /> Analyse
                      </Button>
                    )}
                    {file.status === "Completed" && (
                      <Button size="sm" variant="outline" className="gap-1" onClick={() => watchReport(file)}>
                        View Report
                      </Button>
                    )}
                    {file.status === "Processing" && (
                      <span className="text-xs text-info font-mono">{file.progress}%</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default ArchivePage;

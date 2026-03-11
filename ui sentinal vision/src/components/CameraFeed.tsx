import { Camera } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface Detection {
  label: string;
  confidence: number;
  x: number; y: number; w: number; h: number;
  color: string;
}

interface CameraFeedProps {
  name: string;
  camId: string;
  isActive?: boolean;
  compact?: boolean;
  onDetection?: (label: string, confidence: number) => void;
}


const CameraFeed = ({ name, camId, isActive = true, compact = false, onDetection }: CameraFeedProps) => {
  const [streamError, setStreamError] = useState(false);

  // Ensure camId is consistently used as-is, converted to lowercase to match backend registry keys
  const formattedCamId = camId.toLowerCase();

  const streamUrl = `/video-feed/${formattedCamId}`;


  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden card-hover">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border">
        <div className="flex items-center gap-2">
          <Camera className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">{name}</span>
        </div>
        {isActive && (
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-live live-pulse" />
            <span className="text-xs font-mono text-muted-foreground">LIVE</span>
          </div>
        )}
      </div>
      <div className="relative bg-muted overflow-hidden flex items-center justify-center min-h-[280px]">
        {isActive && !streamError ? (
          <img
            src={streamUrl}
            alt={`Live Feed ${camId}`}
            className="w-full h-auto object-contain"
            onError={() => setStreamError(true)}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-muted">
            <div className="text-center">
              <Camera className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <span className="text-sm text-muted-foreground">
                {streamError ? "Backend Stream Unavailable" : "Connecting..."}
              </span>
            </div>
          </div>
        )}
        <div className="absolute top-2 left-2 bg-card/80 backdrop-blur-sm px-2.5 py-1 rounded-md border border-border">
          <span className="text-xs font-mono text-foreground">{camId}</span>
        </div>
      </div>
    </div>
  );
};

export default CameraFeed;

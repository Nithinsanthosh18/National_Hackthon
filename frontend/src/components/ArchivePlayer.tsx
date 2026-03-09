import React, { useRef, useState, useEffect } from 'react';
import { VideoHeatmap } from './VideoHeatmap';
import AlertsPanel from './AlertsPanel';

interface ArchivePlayerProps {
    videoUrl: string;
    findings?: any[];
    onClose: () => void;
}

export const ArchivePlayer: React.FC<ArchivePlayerProps> = ({ videoUrl, findings = [], onClose }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);

    // interactionData tracks how many times each 1-second segment was viewed
    const [interactionData, setInteractionData] = useState<number[]>([]);

    useEffect(() => {
        // Initialize interaction array when duration is known
        if (duration > 0 && interactionData.length === 0) {
            setInteractionData(new Array(Math.ceil(duration)).fill(0));
        }
    }, [duration]);

    const handleTimeUpdate = () => {
        if (!videoRef.current) return;

        const time = videoRef.current.currentTime;
        setCurrentTime(time);

        // Only track engagement if playing (not scrubbing/paused)
        if (!videoRef.current.paused && duration > 0) {
            const secondIndex = Math.floor(time);
            if (secondIndex >= 0 && secondIndex < interactionData.length) {
                setInteractionData(prev => {
                    const newData = [...prev];
                    newData[secondIndex] += 1;
                    return newData;
                });
            }
        }
    };

    const handleSeek = (time: number) => {
        if (videoRef.current) {
            videoRef.current.currentTime = time;
        }
    };

    return (
        <div className="bg-card border border-border rounded-xl overflow-hidden shadow-lg animate-in fade-in zoom-in-95 duration-200">
            <div className="flex justify-between items-center p-3 border-b border-border bg-muted/20">
                <h3 className="font-medium text-sm flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-success animate-pulse"></span>
                    Activity Review Player
                </h3>
                <button
                    onClick={onClose}
                    className="text-muted-foreground hover:text-foreground hover:bg-muted p-1 rounded transition-colors"
                >
                    ✕
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-0 border-t border-border">
                <div className="p-4 lg:col-span-2 border-r border-border">
                    <video
                        ref={videoRef}
                        src={videoUrl}
                        className="w-full aspect-video bg-black rounded-lg mb-4"
                        controls
                        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                        onTimeUpdate={handleTimeUpdate}
                    />

                    {duration > 0 ? (
                        <VideoHeatmap
                            interactionData={interactionData}
                            duration={duration}
                            currentTime={currentTime}
                            onSeek={handleSeek}
                        />
                    ) : (
                        <div className="h-12 flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded-md">
                            Load video to analyze engagement
                        </div>
                    )}
                </div>

                <div className="p-4 lg:col-span-1 bg-muted/10 overflow-y-auto max-h-[600px] flex flex-col">
                    <h4 className="text-sm font-semibold mb-3 text-foreground flex items-center gap-2">
                        AI Detections Log
                        {findings.length > 0 && (
                            <span className="text-xs font-normal text-muted-foreground bg-accent px-2 py-0.5 rounded-full">{findings.length} events</span>
                        )}
                    </h4>
                    <div className="flex-1">
                        {findings.length > 0 ? (
                            <AlertsPanel alerts={findings} />
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50 p-6 space-y-4">
                                <p className="text-sm text-center">No threats or events detected in this recording.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

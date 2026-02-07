'use client';

import { useState } from 'react';
import { Info, X } from 'lucide-react';

interface InfoTooltipProps {
    title: string;
    content: string;
    className?: string;
}

export function InfoTooltip({ title, content, className = '' }: InfoTooltipProps) {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <div className={`relative inline-block ${className}`}>
            <button
                onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setIsOpen(!isOpen);
                }}
                className="info-trigger"
                aria-label={`Info about ${title}`}
            >
                <Info className="w-3 h-3" />
            </button>

            {isOpen && (
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 z-40"
                        onClick={() => setIsOpen(false)}
                    />

                    {/* Tooltip */}
                    <div className="absolute z-50 left-6 top-0 w-64 p-4 bg-white dark:bg-surface-800 rounded-xl shadow-xl border border-surface-200 dark:border-surface-700 animate-fade-in">
                        <div className="flex items-start justify-between mb-2">
                            <h4 className="font-semibold text-surface-900 dark:text-white">
                                {title}
                            </h4>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="p-1 rounded hover:bg-surface-100 dark:hover:bg-surface-700"
                            >
                                <X className="w-4 h-4 text-surface-500" />
                            </button>
                        </div>
                        <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                            {content}
                        </p>
                    </div>
                </>
            )}
        </div>
    );
}

'use client';

import { useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { createChart, ColorType, IChartApi } from 'lightweight-charts';
import { stocksAPI } from '@/lib/api';

export function PriceChart() {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    const { data: historyData } = useQuery({
        queryKey: ['price-history'],
        queryFn: () => stocksAPI.getHistory('^NSEI', 60),
    });

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Create chart
        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#9ca3af',
            },
            grid: {
                vertLines: { color: 'rgba(156, 163, 175, 0.1)' },
                horzLines: { color: 'rgba(156, 163, 175, 0.1)' },
            },
            width: chartContainerRef.current.clientWidth,
            height: 350,
            rightPriceScale: {
                borderVisible: false,
            },
            timeScale: {
                borderVisible: false,
                timeVisible: true,
            },
            crosshair: {
                vertLine: {
                    labelBackgroundColor: '#6366f1',
                },
                horzLine: {
                    labelBackgroundColor: '#6366f1',
                },
            },
        });

        chartRef.current = chart;

        // Add candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            borderDownColor: '#ef4444',
            borderUpColor: '#10b981',
            wickDownColor: '#ef4444',
            wickUpColor: '#10b981',
        });

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({
            color: '#6366f1',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });

        volumeSeries.priceScale().applyOptions({
            scaleMargins: {
                top: 0.9,
                bottom: 0,
            },
        });

        // Set data if available
        if (historyData?.data) {
            const prices = historyData.data.reverse();

            const candleData = prices.map((p: any) => ({
                time: p.date,
                open: p.open,
                high: p.high,
                low: p.low,
                close: p.close,
            }));

            const volumeData = prices.map((p: any) => ({
                time: p.date,
                value: p.volume || 0,
                color: p.close >= p.open ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)',
            }));

            candlestickSeries.setData(candleData);
            volumeSeries.setData(volumeData);
            chart.timeScale().fitContent();
        }

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [historyData]);

    return (
        <div ref= { chartContainerRef } className = "chart-container" />
  );
}

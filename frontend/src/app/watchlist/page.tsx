'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Plus, Trash2, Star, TrendingUp, TrendingDown } from 'lucide-react';
import { usersAPI, stocksAPI } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface WatchlistItem {
    id: number;
    symbol: string;
    display_name: string | null;
    created_at: string;
}

const AVAILABLE_STOCKS = [
    { symbol: '^NSEI', name: 'NIFTY 50 Index' },
    { symbol: 'RELIANCE.NS', name: 'Reliance Industries' },
    { symbol: 'TCS.NS', name: 'Tata Consultancy Services' },
    { symbol: 'HDFCBANK.NS', name: 'HDFC Bank' },
    { symbol: 'INFY.NS', name: 'Infosys' },
    { symbol: 'ICICIBANK.NS', name: 'ICICI Bank' },
    { symbol: 'HINDUNILVR.NS', name: 'Hindustan Unilever' },
    { symbol: 'SBIN.NS', name: 'State Bank of India' },
    { symbol: 'BAJFINANCE.NS', name: 'Bajaj Finance' },
    { symbol: 'ITC.NS', name: 'ITC Limited' },
];

export default function WatchlistPage() {
    const queryClient = useQueryClient();
    const [showAdd, setShowAdd] = useState(false);

    const { data: watchlistData, isLoading } = useQuery({
        queryKey: ['watchlist'],
        queryFn: () => usersAPI.getWatchlist(),
    });

    const addMutation = useMutation({
        mutationFn: (symbol: string) => usersAPI.addToWatchlist(symbol),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
            setShowAdd(false);
        },
    });

    const removeMutation = useMutation({
        mutationFn: (id: number) => usersAPI.removeFromWatchlist(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
        },
    });

    const watchlist: WatchlistItem[] = watchlistData?.data || [];
    const watchedSymbols = new Set(watchlist.map((w) => w.symbol));

    return (
        <div className= "space-y-6" >
        {/* Header */ }
        < div className = "flex items-center justify-between" >
            <div>
            <h1 className="text-3xl font-bold text-surface-900 dark:text-white" > Watchlist </h1>
                < p className = "text-surface-500 mt-1" > Track your favorite stocks </p>
                    </div>
                    < div className = "flex items-center gap-2" >
                        <InfoTooltip
            title="Watchlist"
    content = "Add stocks to your watchlist to quickly access their prices and predictions. You can also set price alerts for watchlist items."
        />
        <button
            onClick={ () => setShowAdd(true) }
    className = "flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
        >
        <Plus className="w-5 h-5" />
            Add Stock
                </button>
                </div>
                </div>

    {/* Add Stock Modal */ }
    {
        showAdd && (
            <div className="card p-6" >
                <h3 className="text-lg font-semibold mb-4" > Add to Watchlist </h3>
                    < div className = "grid grid-cols-1 md:grid-cols-2 gap-3" >
                    {
                        AVAILABLE_STOCKS.filter((s) => !watchedSymbols.has(s.symbol)).map((stock) => (
                            <button
                key= { stock.symbol }
                onClick = {() => addMutation.mutate(stock.symbol)}
        className = "flex items-center justify-between p-4 rounded-lg border border-surface-200 dark:border-surface-600 hover:bg-surface-50 dark:hover:bg-surface-700 transition-colors text-left"
            >
            <div>
            <p className="font-medium text-surface-900 dark:text-white" > { stock.name } </p>
                < p className = "text-sm text-surface-500" > { stock.symbol } </p>
                    </div>
                    < Plus className = "w-5 h-5 text-primary-500" />
                        </button>
            ))
    }
    </div>
        < button
    onClick = {() => setShowAdd(false)
}
className = "mt-4 px-4 py-2 border border-surface-200 dark:border-surface-600 rounded-lg hover:bg-surface-50 dark:hover:bg-surface-700"
    >
    Close
    </button>
    </div>
      )}

{/* Watchlist */ }
{
    isLoading ? (
        <div className= "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" >
        {
            [1, 2, 3].map((i) => (
                <div key= { i } className = "card p-6" >
                <div className="skeleton h-6 w-3/4 mb-3" />
            <div className="skeleton h-8 w-1/2 mb-2" />
            <div className="skeleton h-4 w-1/3" />
            </div>
            ))
        }
        </div>
      ) : watchlist.length === 0 ? (
        <div className= "card p-12 text-center" >
        <Star className="w-12 h-12 mx-auto text-surface-300 mb-4" />
            <h3 className="text-lg font-medium text-surface-700 dark:text-surface-300" > No stocks in watchlist </h3>
                < p className = "text-surface-500 mt-1" > Add stocks to track their prices.</p>
                    </div>
      ) : (
        <div className= "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" >
        {
            watchlist.map((item) => (
                <WatchlistCard
              key= { item.id }
              item = { item }
              onRemove = {() => removeMutation.mutate(item.id)}
        />
          ))
}
</div>
      )}
</div>
  );
}

function WatchlistCard({ item, onRemove }: { item: WatchlistItem; onRemove: () => void }) {
    const { data, isLoading } = useQuery({
        queryKey: ['realtime', item.symbol],
        queryFn: () => stocksAPI.getRealtime(item.symbol),
        refetchInterval: 60000,
    });

    const price = data?.data;
    const isPositive = price?.change_pct >= 0;
    const stockName = AVAILABLE_STOCKS.find((s) => s.symbol === item.symbol)?.name || item.symbol;

    return (
        <div className= "card p-6 card-hover" >
        <div className="flex items-start justify-between mb-3" >
            <div>
            <h3 className="font-semibold text-surface-900 dark:text-white" > { stockName } </h3>
                < p className = "text-sm text-surface-500" > { item.symbol } </p>
                    </div>
                    < button onClick = { onRemove } className = "text-surface-400 hover:text-danger-500 transition-colors" >
                        <Trash2 className="w-4 h-4" />
                            </button>
                            </div>

    {
        isLoading ? (
            <div className= "skeleton h-8 w-24" />
      ) : (
            <>
            <p className= "text-2xl font-bold text-surface-900 dark:text-white" >
            ₹{ price?.price?.toLocaleString('en-IN') || '—' }
        </p>
            < div className = {`flex items-center gap-1 mt-1 ${isPositive ? 'positive' : 'negative'}`
    }>
        { isPositive?<TrendingUp className = "w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
<span className="text-sm font-medium" >
    { isPositive? '+': '' }{ price?.change_pct?.toFixed(2) }%
        </span>
        </div>
        </>
      )}
</div>
  );
}

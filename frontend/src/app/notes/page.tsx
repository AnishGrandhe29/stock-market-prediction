'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Plus, Trash2, Search, FileText } from 'lucide-react';
import { usersAPI } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface Note {
    id: number;
    title: string | null;
    content: string;
    symbol: string | null;
    tags: string | null;
    created_at: string;
    updated_at: string;
}

export default function NotesPage() {
    const queryClient = useQueryClient();
    const [isCreating, setIsCreating] = useState(false);
    const [newNote, setNewNote] = useState({ title: '', content: '', symbol: '' });
    const [search, setSearch] = useState('');

    const { data: notesData, isLoading } = useQuery({
        queryKey: ['notes'],
        queryFn: () => usersAPI.getNotes(),
    });

    const createMutation = useMutation({
        mutationFn: (data: { title?: string; content: string; symbol?: string }) =>
            usersAPI.createNote(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['notes'] });
            setIsCreating(false);
            setNewNote({ title: '', content: '', symbol: '' });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: (id: number) => usersAPI.deleteNote(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['notes'] });
        },
    });

    const notes: Note[] = notesData?.data || [];
    const filteredNotes = notes.filter(
        (n) =>
            n.title?.toLowerCase().includes(search.toLowerCase()) ||
            n.content.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className= "space-y-6" >
        {/* Header */ }
        < div className = "flex items-center justify-between" >
            <div>
            <h1 className="text-3xl font-bold text-surface-900 dark:text-white" > Notes </h1>
                < p className = "text-surface-500 mt-1" > Your personal trading journal </p>
                    </div>
                    < div className = "flex items-center gap-2" >
                        <InfoTooltip
            title="Notes"
    content = "Keep track of your trading ideas, observations, and research. Notes are private and only visible to you."
        />
        <button
            onClick={ () => setIsCreating(true) }
    className = "flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
        >
        <Plus className="w-5 h-5" />
            New Note
                </button>
                </div>
                </div>

    {/* Search */ }
    <div className="relative max-w-md" >
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-surface-400" />
            <input
          type="text"
    value = { search }
    onChange = {(e) => setSearch(e.target.value)
}
placeholder = "Search notes..."
className = "w-full pl-10 pr-4 py-2 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700 text-surface-900 dark:text-white placeholder-surface-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
    />
    </div>

{/* Create Note Modal */ }
{
    isCreating && (
        <div className="card p-6" >
            <h3 className="text-lg font-semibold mb-4" > Create New Note </h3>
                < div className = "space-y-4" >
                    <input
              type="text"
    value = { newNote.title }
    onChange = {(e) => setNewNote({ ...newNote, title: e.target.value })
}
placeholder = "Note title (optional)"
className = "w-full px-4 py-2 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700"
    />
    <textarea
              value={ newNote.content }
onChange = {(e) => setNewNote({ ...newNote, content: e.target.value })}
placeholder = "Write your note..."
rows = { 4}
className = "w-full px-4 py-2 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700 resize-none"
    />
    <input
              type="text"
value = { newNote.symbol }
onChange = {(e) => setNewNote({ ...newNote, symbol: e.target.value })}
placeholder = "Related stock symbol (optional, e.g., ^NSEI)"
className = "w-full px-4 py-2 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700"
    />
    <div className="flex gap-2" >
        <button
                onClick={
    () => createMutation.mutate({
        title: newNote.title || undefined,
        content: newNote.content
    })
}
disabled = {!newNote.content}
className = "px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50"
    >
    Save Note
        </button>
        < button
onClick = {() => setIsCreating(false)}
className = "px-4 py-2 border border-surface-200 dark:border-surface-600 rounded-lg hover:bg-surface-50 dark:hover:bg-surface-700"
    >
    Cancel
    </button>
    </div>
    </div>
    </div>
      )}

{/* Notes Grid */ }
{
    isLoading ? (
        <div className= "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" >
        {
            [1, 2, 3].map((i) => (
                <div key= { i } className = "card p-6" >
                <div className="skeleton h-6 w-3/4 mb-3" />
            <div className="skeleton h-4 w-full mb-2" />
            <div className="skeleton h-4 w-2/3" />
            </div>
            ))
        }
        </div>
      ) : filteredNotes.length === 0 ? (
        <div className= "card p-12 text-center" >
        <FileText className="w-12 h-12 mx-auto text-surface-300 mb-4" />
            <h3 className="text-lg font-medium text-surface-700 dark:text-surface-300" > No notes yet </h3>
                < p className = "text-surface-500 mt-1" > Create your first note to start journaling.</p>
                    </div>
      ) : (
        <div className= "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" >
        {
            filteredNotes.map((note) => (
                <div key= { note.id } className = "card p-6 card-hover" >
                <div className="flex items-start justify-between mb-2" >
            <h3 className="font-semibold text-surface-900 dark:text-white truncate" >
            { note.title || 'Untitled Note' }
            </h3>
            < button
                  onClick = {() => deleteMutation.mutate(note.id)}
    className = "text-surface-400 hover:text-danger-500 transition-colors"
        >
        <Trash2 className="w-4 h-4" />
            </button>
            </div>
            < p className = "text-sm text-surface-600 dark:text-surface-400 line-clamp-3" >
                { note.content }
                </p>
    {
        note.symbol && (
            <span className="inline-block mt-3 px-2 py-1 text-xs bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 rounded" >
                { note.symbol }
                </span>
              )
    }
    <p className="text-xs text-surface-400 mt-3" >
        { new Date(note.updated_at).toLocaleDateString() }
        </p>
        </div>
          ))
}
</div>
      )}
</div>
  );
}

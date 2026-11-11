// Runs before first paint (a classic script, not a module) so a saved light theme never flashes dark.
try { const t = localStorage.getItem('apo-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }

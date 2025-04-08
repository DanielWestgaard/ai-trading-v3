class PositionManager:
    def __init__(self, min_hold_bars=12, max_hold_bars=288):  # 1 hour to 1 day
        self.min_hold_bars = min_hold_bars
        self.max_hold_bars = max_hold_bars
        self.position_tracking = {}  # symbol -> position info
        
    def on_bar(self, symbol, timestamp):
        """Update tracking for each new bar."""
        if symbol in self.position_tracking and self.position_tracking[symbol]['active']:
            self.position_tracking[symbol]['bars_held'] += 1
    
    def open_position(self, symbol, timestamp, direction):
        """Record a new position opening."""
        self.position_tracking[symbol] = {
            'active': True,
            'entry_time': timestamp,
            'direction': direction,
            'bars_held': 0
        }
    
    def close_position(self, symbol, timestamp):
        """Record position closing."""
        if symbol in self.position_tracking:
            self.position_tracking[symbol]['active'] = False
    
    def can_exit(self, symbol):
        """Check if a position can be exited based on holding time."""
        if symbol not in self.position_tracking or not self.position_tracking[symbol]['active']:
            return False
            
        bars_held = self.position_tracking[symbol]['bars_held']
        return bars_held >= self.min_hold_bars
    
    def should_exit(self, symbol):
        """Check if a position should be force-exited (max holding time)."""
        if symbol not in self.position_tracking or not self.position_tracking[symbol]['active']:
            return False
            
        bars_held = self.position_tracking[symbol]['bars_held']
        return bars_held >= self.max_hold_bars
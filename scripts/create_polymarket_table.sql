CREATE TABLE IF NOT EXISTS polymarket_portfolio (
  id integer PRIMARY KEY DEFAULT 1,
  positions jsonb DEFAULT '{}',
  resolved jsonb DEFAULT '[]',
  total_invested float DEFAULT 0,
  current_value float DEFAULT 0,
  unrealized_pnl float DEFAULT 0,
  realized_pnl float DEFAULT 0,
  updated_at timestamptz DEFAULT now()
);
ALTER TABLE polymarket_portfolio ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS anon_select_polymarket ON polymarket_portfolio;
CREATE POLICY anon_select_polymarket ON polymarket_portfolio FOR SELECT TO anon USING (true);

CREATE TABLE valuations (
    ticker VARCHAR(10) PRIMARY KEY,
    company VARCHAR(100),
    current_price DECIMAL(12, 2),
    currency VARCHAR(10),
    base_revenue DECIMAL(20, 2),
    shares_outstanding DECIMAL(20, 2),
    
    -- Relational Summary Stats
    expected_value DECIMAL(12, 2), 
    expected_cagr DECIMAL(8, 4),
    
    -- Dispersion Metric: (max IV - min IV) / Expected Value
    dispersion_ratio DECIMAL(8, 4), 
    
    recommendation VARCHAR(50),

    -- JSONB Payloads
    thesis_data JSONB NOT NULL,
    valuation_data JSONB NOT NULL, -- Stores the full scenario inputs/outputs
    
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_valuations_thesis ON valuations USING GIN (thesis_data);
CREATE INDEX idx_valuations_model ON valuations USING GIN (valuation_data);

-- Index the spread to find "Stable" vs "Speculative" valuations
CREATE INDEX idx_valuations_dispersion ON valuations (dispersion_ratio);
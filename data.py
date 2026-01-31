import duckdb
import os

SCALE_FACTORS = [0.5, 1, 2, 5]
TABLES = ['region', 'nation', 'supplier', 'part', 
          'partsupp', 'customer', 'orders', 'lineitem']

def generate_data(scale_factor, output_dir='data'):
    # Generate TPC-H data for a given scale factor
    os.makedirs(output_dir, exist_ok=True)

    conn = duckdb.connect()
    conn.execute()
    conn.execute("INSTALL tpch; LOAD tpch;")
    conn.execute(f"CALL tpch.sf{scale_factor}();")
                 
    for table in TABLES:
        path =os.path.join(output_dir, f"{table}.parquet")
        conn.execute(f"COPY {table} TO '{path}' (FORMAT 'parquet');")
        print(f"  Exported {table} to {path}")
    
    conn.close()

if __name__ == "__main__":
    for sf in SCALE_FACTORS:
        print(f"Generating SF={sf}...")
        generate_data(sf, f"data/sf{sf}")
use arrow::record_batch::RecordBatch;
use deltalake::datafusion::prelude::SessionContext;
use deltalake::open_table;
use futures::executor::block_on;
use polars::lazy::dsl::Expr;
use polars::lazy::dsl::*;
use polars::lazy::dsl::*;
use polars::prelude::*;
use polars::prelude::*;
use polars::series::ops::NullBehavior;
use polars_plan::prelude::*;
use std::sync::Arc;
use tokio;

// use polars::lazy::dsl::Expr;// Because polars uses a forked backend of arrow,
// and since type conversion has not been straightforward,
// the arrow to polars_arrow conversion is done by serialisation and deserialisation :
// Arrow -> Parquet -> Polars
// Could also try with IPC support, or conversion to C FFI implementation
fn arrow_to_polars(batch: &RecordBatch) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Write Arrow batch to parquet file
    let mut buffer = Vec::new();
    {
        use parquet::arrow::ArrowWriter;
        let schema = batch.schema();
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None)?;
        writer.write(batch)?;
        writer.close()?;
    }

    // Read parquet into Polars
    let cursor = std::io::Cursor::new(buffer);
    let df = polars::prelude::ParquetReader::new(cursor).finish()?;
    Ok(df)
}

// load the given delta table to polars dataframe
fn delta_to_polars(delta_table_path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Read and open the delta table
    let table = block_on(async { open_table(delta_table_path).await })?;

    println!("Delta table loaded successfully.");

    // Create a DataFusion context and register the table
    let ctx = SessionContext::new();
    let table_ref = Arc::new(table);

    // Register the Delta table with DataFusion
    block_on(async { ctx.register_table("delta_table", table_ref.clone()) })?;

    let df = block_on(async {
        let query_result = ctx.sql("SELECT * FROM delta_table").await?;
        // Execute the query and collect results
        let batches = query_result.collect().await?;

        if batches.is_empty() {
            return Err("Delta table contains no data".into());
        }
        // Convert record batches to Polars DataFrame
        let mut all_dfs = Vec::new();
        for batch in &batches {
            let df = arrow_to_polars(batch).unwrap();
            all_dfs.push(df);
        }
        print!("{}", all_dfs.get(0).unwrap());

        // Concatenate all dataframes
        if all_dfs.len() == 1 {
            Ok::<DataFrame, Box<dyn std::error::Error>>(all_dfs.remove(0))
        } else {
            let mut result = all_dfs.remove(0);

            // Stack the remaining DataFrames onto the first one
            for df in all_dfs {
                result = result.vstack(&df).expect("Failed to stack DataFrame");
            }
            Ok::<DataFrame, Box<dyn std::error::Error>>(result)
        }
        // Ok(())
    })?;

    Ok(df)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let delta_table_path = "../../raw_data";
    const MAX_DAILY_CHANGE: f32 = 2.0;

    // Read the Delta table and convert to Polars
    let mut polars_df = delta_to_polars(delta_table_path)?;

    // logic to filter jumps
    polars_df = polars_df
        .lazy()
        .select([all().exclude(["time"]).backward_fill(None)])
        .with_columns([all().exclude(["time"]).first().alias("__first_row")])
        .with_columns([all()
            .exclude(["time", "__first_row"])
            .diff(1, NullBehavior::Drop)
            .clip((-MAX_DAILY_CHANGE).into(), MAX_DAILY_CHANGE.into())
            .replace(-MAX_DAILY_CHANGE, 0.)
            .replace(MAX_DAILY_CHANGE, 0.)
            .cum_sum(false)
            + col("__first_row")])
        .drop(["__first_row"])
        .collect()?;

    println!("Polars DataFrame Head:");
    println!("{}", polars_df.head(Some(5)));

    Ok(())
}

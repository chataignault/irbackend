use deltalake::{DeltaTable, DeltaTableBuilder, DeltaTableError, open_table, writer::DeltaWriter};
use polars::prelude::*;
use arrow::prelude::*;
use std::path::Path;
use tokio;
use std::sync::Arc;
use deltalake::datafusion::execution::context::SessionContext;

#[tokio::main]
async fn main() -> Result<(), DeltaTableError> {

    let table_path = "../../raw_data";
    
    // open the table written in python
    let table = open_table(table_path).await?;
    
    // show all active files in the table
    let files: Vec<_> = table.get_file_uris()?.collect();
    let schema = table.schema().unwrap();
    println!("{files:?}");
    println!("{schema:?}");
    
    
    // _ = table.load();
    println!("Loaded table");

    
    let batches = async {
        let mut ctx = SessionContext::new();
        let table = open_table(table_path)
            .await
            .unwrap();
        ctx.register_table("demo", Arc::new(table)).unwrap();
    
        let batches = ctx
            .sql("SELECT * FROM demo").await.unwrap()
            .collect()
            .await.unwrap();
        batches
    }.await;
    let batch = batches.get(0).unwrap();
    let arrow_schema = batch.schema();
    let ipc_data = arrow::ipc::writer::serialize_batch(batch)?;
    
    // Then deserialize it using polars' native arrow implementation
    let record_batch = arrow::ipc::reader::read_record_batch(&ipc_data, arrow_schema.clone(), 0)?;
    
    // Now convert the standard Arrow RecordBatch to a polars DataFrame
    let df = DataFrame::try_from(record_batch).unwrap();
    // let df = DataFrame::try_from(batch).unwrap();
    Ok(())
}

use deltalake::{DeltaTableBuilder, DeltaTableError, open_table, writer::DeltaWriter};
use polars::prelude::*;
use std::path::Path;
use tokio;

#[tokio::main]
async fn main() -> Result<(), DeltaTableError> {
    let id_series = Series::new("id".into(), &[1_i32, 2_i32]);
    let value_series = Series::new("value".into(), &["foo", "boo"]);

    let df = DataFrame::new(vec![id_series.into(), value_series.into()])?;
    println!("Original DataFrame:");
    println!("{}", df);

    // Write the DataFrame to a Delta table
    let table_path = "./data/delta";
    let writer = DeltaWriter::with_writer_properties();
    writer::Writer(Path::new(table_path), &df, None, None, None)?;

    // Read back the Delta table
    let dt = DeltaTableBuilder::from_uri(table_path).load()?;

    // Convert the Delta table to a Polars DataFrame
    let df2 = deltalake::to_polars(&dt)?;
    println!("DataFrame from Delta:");
    println!("{}", df2);

    // open the table written in python
    let table = open_table(table_path).await?;

    // show all active files in the table
    let files: Vec<_> = table.get_file_uris()?.collect();
    println!("{files:?}");

    Ok(())
}

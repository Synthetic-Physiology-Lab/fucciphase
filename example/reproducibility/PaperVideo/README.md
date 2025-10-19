Process a TrackMate file through the command-line interface (CLI):

```
fucciphase merged_linked.ome.xml -ref ../../example_data/hacat_fucciphase_reference.csv -dt 0.25 -m MEAN_INTENSITY_CH1 -c MEAN_INTENSITY_CH2 --generate_unique_tracks true
```

To get more info, run
```
fucciphase -h
```

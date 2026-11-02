# SURA: 1 TX + 4 RX synchronised Wi-Fi CSI

## How synchronisation works (no clock sync needed)
1. The TX board broadcasts a numbered ESP-NOW packet at 100 Hz on a fixed channel: `(boot_id, seq, tx_us)`.
2. Every RX board hears the same packets. Its sniffer reads `seq` from the packet; its CSI callback gives
   the CSI of that same packet; the two are paired by the identical hardware RX timestamp.
3. The RX streams `(rx_id, tx_boot, tx_seq, tx_us, rx_ts_us, rssi, flags, raw CSI)` as CRC-checked binary frames.
4. Offline, `merge_session.py` joins all receivers on `(tx_boot, tx_seq)`. Receivers on different laptops align
   exactly, even with unrelated clocks, because the key is the TX counter.
5. The camera is tied in through the laptop it is plugged into: `record_camera.py` and `record_session.py` use the same
   `time.monotonic()`; `merge_session.py` fits host time vs TX time (lower envelope, so bursty USB latency is
   ignored) and converts camera frames, markers and trial start/end to TX time/seq. A visible flash at a marker
   removes the camera delivery lag.

The TX needs only USB power, so it does not occupy a laptop port. With 2-3 ports per laptop use 2+2 or 3+1
receivers on two laptops (or one powered hub). Nothing in the design depends on the split.

## Setup
- Firmware: `pio run -e tx -t upload`; `-e rx1 ... rx4` for the four receivers (one board each; `RX_ID` is baked in).
  Change `WIFI_CH` in `platformio.ini` to the least busy 2.4 GHz channel (check with a phone Wi-Fi analyser). Keep TX and RX on the same channel.
- Host: `pip install pyserial numpy opencv-python`.
- Session (on each laptop): `python host/record_session.py --ports <rx ports> --out sessions/<name>_<laptop>`.
  Camera laptop also: `python host/record_camera.py --out sessions/<name>_<camera laptop>`.
  Console: `m` marker (do the visible flash at that instant), `s <label>` trial start, `e` trial end, `n <text>`, `q`.
- Merge: `python host/merge_session.py <dirA> <dirB> --out merged/<name>.npz --flash-frames <frame idx of each marker's flash>`
- No hardware? `python host/selftest_sim.py` (passes: exact re-alignment across two simulated laptops with packet loss,
  different clocks, a 32-bit TX timer wrap and a 60 ms camera lag).

## Things to know
- ESP-NOW defaults to 1 Mbps 802.11b which carries no LTF, so the TX forces MCS0 (HT). Packet type decides the CSI
  length: 128/256/384 bytes (LLTF / +HT-LTF / +STBC-HT-LTF). A plain ESP-NOW packet is usually 256 bytes, not the 384 your
  AP+ping setup gave. Don't mix datasets of different length. `merge_session.py` prints the length histogram.
- Raw bytes are `[imag, real]` per subcarrier (ESP-IDF docs); `csi_to_complex()` does this. On ESP32-S3 the first 4 bytes are invalid.
- Use `tx_seq`/`tx_t` as the time axis: uniform ~100 Hz across all receivers. `tx_us` is stamped just before send, so on-air time jitters by ~ms.
- Check after every first run: RX status lines show `matched` rising; `orphan_promisc` and `unmatched` near 0.

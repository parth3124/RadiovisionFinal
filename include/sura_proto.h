// Shared wire formats for the SURA 1-TX / N-RX CSI system.
#pragma once
#include <stdint.h>
#include <stddef.h>

static const uint8_t SURA_MAGIC[4] = {'S', 'U', 'R', 'A'};
#define TX_HZ 100

#pragma pack(push, 1)
// Sent over the air by the TX board (ESP-NOW broadcast). Every RX hears the same
// (boot_id, seq), which is what lets us align the receivers without clock sync.
struct TxPayload {
  uint8_t  magic[4];
  uint32_t boot_id;   // random per TX boot: detects a TX reset mid-session
  uint32_t seq;       // increments by 1 per transmitted packet
  uint32_t tx_us;     // low 32 bits of TX esp_timer at send time
};

// RX -> host frame:  FrameHdr | body[csi_len] | crc16(hdr+body)
enum { FT_CSI = 1, FT_STATS = 2 };
enum {                       // FrameHdr.flags
  FL_SIGMODE_MASK = 0x03,    // 0 non-HT, 1 HT, 3 VHT
  FL_CWB40        = 0x04,
  FL_STBC         = 0x08,
  FL_FIRST_WORD_INVALID = 0x10,
  FL_UNMATCHED    = 0x20     // CSI with no TX seq found (debug builds only)
};
struct FrameHdr {
  uint8_t  m0, m1;           // 0xA5 0x5A
  uint8_t  type;
  uint8_t  rx_id;
  uint16_t csi_len;          // body length
  uint8_t  flags;
  int8_t   rssi;
  uint32_t tx_boot;
  uint32_t tx_seq;
  uint32_t tx_us;
  uint32_t rx_ts_us;         // RX hardware timestamp (rx_ctrl.timestamp), local clock
};
struct StatsBody {           // body of FT_STATS (cumulative counters)
  uint32_t matched;          // CSI frames paired with a TX seq and forwarded
  uint32_t unmatched_csi;    // CSI frames with no TX seq (other traffic / pairing failure)
  uint32_t orphan_promisc;   // TX packets seen by sniffer but no CSI produced
  uint32_t drops;            // dropped because USB serial was full
};
#pragma pack(pop)

static inline uint16_t sura_crc16(const uint8_t* d, size_t n) {  // CRC-16/CCITT-FALSE
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)d[i] << 8;
    for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
  }
  return crc;
}

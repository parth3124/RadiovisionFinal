// SURA RX: passive receiver. Sniffs the TX broadcast (promiscuous callback gives the
// TX seq), gets CSI for the same packet (CSI callback), pairs the two by the identical
// hardware RX timestamp, and streams binary frames to the host over USB CDC.
#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include "sura_proto.h"

#ifndef RX_ID
#define RX_ID 2
#endif
#ifndef WIFI_CH
#define WIFI_CH 1
#endif
#define CSI_MAX 612
#define TS_TOL_US 300          // pairing tolerance on rx_ctrl.timestamp
#define WAIT_MS 60             // how long an unpaired item waits for its partner
// #define ALLOW_UNMATCHED     // debug: forward CSI even if no TX seq was found

struct PromEv { uint32_t ts, boot, seq, txus; };
struct CsiEv  { uint32_t ts; int8_t rssi; uint8_t flags; uint16_t len; int8_t buf[CSI_MAX]; };

static QueueHandle_t promQ, csiQ;
static volatile uint32_t drops = 0;
static uint32_t matched = 0, unmatched = 0, orphan = 0;

static inline bool near_ts(uint32_t a, uint32_t b) {
  int32_t d = (int32_t)(a - b);
  return d >= -TS_TOL_US && d <= TS_TOL_US;
}

// ---- Sniffer: find our magic anywhere in the first ~160 bytes of the frame ----
static void promisc_cb(void* buf, wifi_promiscuous_pkt_type_t type) {
  if (type != WIFI_PKT_MGMT && type != WIFI_PKT_DATA) return;
  const wifi_promiscuous_pkt_t* p = (const wifi_promiscuous_pkt_t*)buf;
  uint32_t n = p->rx_ctrl.sig_len;
  if (n > 160) n = 160;
  const uint8_t* d = p->payload;
  for (uint32_t i = 24; i + sizeof(TxPayload) <= n; i++) {
    if (d[i] == 'S' && memcmp(d + i, SURA_MAGIC, 4) == 0) {
      TxPayload t;
      memcpy(&t, d + i, sizeof(t));
      PromEv e = {p->rx_ctrl.timestamp, t.boot_id, t.seq, t.tx_us};
      xQueueSend(promQ, &e, 0);
      return;
    }
  }
}

static void csi_cb(void* ctx, wifi_csi_info_t* info) {
  if (!info || !info->buf || info->len <= 0 || info->len > CSI_MAX) return;
  static CsiEv e;  // callback is single-threaded; xQueueSend copies
  e.ts = info->rx_ctrl.timestamp;
  e.rssi = info->rx_ctrl.rssi;
  e.len = info->len;
  e.flags = (info->rx_ctrl.sig_mode & 3) | (info->rx_ctrl.cwb ? FL_CWB40 : 0) |
            (info->rx_ctrl.stbc ? FL_STBC : 0);
#if CONFIG_IDF_TARGET_ESP32S3
  if (info->first_word_invalid) e.flags |= FL_FIRST_WORD_INVALID;
#endif
  memcpy(e.buf, info->buf, info->len);
  if (xQueueSend(csiQ, &e, 0) != pdTRUE) drops++;
}

// ---- Output ----
static uint8_t outbuf[sizeof(FrameHdr) + CSI_MAX + 2];
static void emit(uint8_t type, uint8_t flags, int8_t rssi, uint32_t boot, uint32_t seq,
                 uint32_t txus, uint32_t rxts, const uint8_t* body, uint16_t len) {
  size_t total = sizeof(FrameHdr) + len + 2;
  if ((size_t)Serial.availableForWrite() < total) { drops++; return; }  // never write partial frames
  FrameHdr h = {0xA5, 0x5A, type, RX_ID, len, flags, rssi, boot, seq, txus, rxts};
  memcpy(outbuf, &h, sizeof(h));
  if (len) memcpy(outbuf + sizeof(h), body, len);
  uint16_t crc = sura_crc16(outbuf, sizeof(h) + len);
  memcpy(outbuf + sizeof(h) + len, &crc, 2);
  Serial.write(outbuf, total);
}
static void emit_csi(const CsiEv& c, bool have, const PromEv& p) {
  uint8_t fl = c.flags | (have ? 0 : FL_UNMATCHED);
  emit(FT_CSI, fl, c.rssi, have ? p.boot : 0, have ? p.seq : 0xFFFFFFFF, have ? p.txus : 0,
       c.ts, (const uint8_t*)c.buf, c.len);
}

// ---- Pairing task: both callbacks may fire in either order ----
static void pair_task(void*) {
  const int PN = 16, CN = 6;
  static PromEv pring[PN]; static uint32_t ptime[PN]; static bool pused[PN];
  static CsiEv cwait[CN];  static uint32_t ctime[CN]; static bool cused[CN];
  static CsiEv c; PromEv p;
  uint32_t last_stats = 0;
  for (;;) {
    bool idle = true;
    while (xQueueReceive(promQ, &p, 0) == pdTRUE) {
      idle = false;
      bool paired = false;
      for (int i = 0; i < CN && !paired; i++)
        if (cused[i] && near_ts(cwait[i].ts, p.ts)) {
          emit_csi(cwait[i], true, p); matched++; cused[i] = false; paired = true;
        }
      if (!paired) {
        int s = 0; uint32_t oldest = 0xFFFFFFFF;
        for (int i = 0; i < PN; i++) { if (!pused[i]) { s = i; oldest = 0; break; }
          if (ptime[i] < oldest) { oldest = ptime[i]; s = i; } }
        pring[s] = p; ptime[s] = millis(); pused[s] = true;
      }
    }
    while (xQueueReceive(csiQ, &c, 0) == pdTRUE) {
      idle = false;
      bool paired = false;
      for (int i = 0; i < PN && !paired; i++)
        if (pused[i] && near_ts(pring[i].ts, c.ts)) {
          emit_csi(c, true, pring[i]); matched++; pused[i] = false; paired = true;
        }
      if (!paired) {
        int s = -1;
        for (int i = 0; i < CN; i++) if (!cused[i]) { s = i; break; }
        if (s < 0) { s = 0; unmatched++; }     // overwrite oldest: counted as unmatched
        cwait[s] = c; ctime[s] = millis(); cused[s] = true;
      }
    }
    uint32_t now = millis();
    for (int i = 0; i < CN; i++)
      if (cused[i] && now - ctime[i] > WAIT_MS) {
        cused[i] = false; unmatched++;
#ifdef ALLOW_UNMATCHED
        PromEv z = {0, 0, 0xFFFFFFFF, 0}; emit_csi(cwait[i], false, z);
#endif
      }
    for (int i = 0; i < PN; i++)
      if (pused[i] && now - ptime[i] > WAIT_MS) { pused[i] = false; orphan++; }
    if (now - last_stats >= 1000) {
      last_stats = now;
      StatsBody s = {matched, unmatched, orphan, drops};
      emit(FT_STATS, 0, 0, 0, 0, 0, 0, (const uint8_t*)&s, sizeof(s));
    }
    if (idle) vTaskDelay(1);
  }
}

void setup() {
  Serial.setTxBufferSize(32768);
  Serial.begin(115200);               // baud is ignored on native USB CDC
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  Serial.setTxTimeoutMs(0);           // never block when the host is not reading
#endif
  delay(300);
  promQ = xQueueCreate(64, sizeof(PromEv));
  csiQ  = xQueueCreate(16, sizeof(CsiEv));

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);

  wifi_promiscuous_filter_t filt = {.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA};
  esp_wifi_set_promiscuous_filter(&filt);
  esp_wifi_set_promiscuous_rx_cb(&promisc_cb);
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(WIFI_CH, WIFI_SECOND_CHAN_NONE);

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;
  cfg.htltf_en = true;
  cfg.stbc_htltf2_en = true;
  cfg.ltf_merge_en = true;
  cfg.channel_filter_en = false;
  cfg.manu_scale = false;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(&csi_cb, NULL);
  esp_wifi_set_csi(true);

  xTaskCreatePinnedToCore(pair_task, "pair", 8192, NULL, 5, NULL, 1);
}

void loop() { delay(1000); }

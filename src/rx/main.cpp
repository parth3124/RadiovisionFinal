// SURA RX: captures TX metadata and CSI, pairs them using RX hardware
// timestamps, queues complete frames, and uses a separate task for USB CDC.
//
// FrameHdr and StatsBody wire protocol remain unchanged.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include "sura_proto.h"

#ifndef RX_ID
#define RX_ID 1
#endif

#ifndef WIFI_CH
#define WIFI_CH 1
#endif

#define CSI_MAX 612
#define TS_TOL_US 300
#define WAIT_MS 60

#define PROM_Q_LEN 128
#define CSI_Q_LEN 48
#define OUT_Q_LEN 32

#define PROM_PENDING 64
#define CSI_PENDING 16

struct PromEv {
    uint32_t ts, boot, seq, txus;
};

struct CsiEv {
    uint32_t ts;
    int8_t rssi;
    uint8_t flags;
    uint16_t len;
    int8_t buf[CSI_MAX];
};

#define MAX_FRAME (sizeof(FrameHdr) + CSI_MAX + 2)

struct OutFrame {
    uint16_t len;
    uint8_t bytes[MAX_FRAME];
};

static QueueHandle_t promQ;
static QueueHandle_t csiQ;
static QueueHandle_t outQ;

static portMUX_TYPE statsMux = portMUX_INITIALIZER_UNLOCKED;

static volatile uint32_t matched = 0;
static volatile uint32_t unmatched = 0;
static volatile uint32_t orphan = 0;
static volatile uint32_t drops = 0;


// -------------------- Statistics --------------------

static void addDrop() {
    portENTER_CRITICAL(&statsMux);
    drops++;
    portEXIT_CRITICAL(&statsMux);
}

static void addUnmatched() {
    portENTER_CRITICAL(&statsMux);
    unmatched++;
    portEXIT_CRITICAL(&statsMux);
}

static void addOrphan() {
    portENTER_CRITICAL(&statsMux);
    orphan++;
    portEXIT_CRITICAL(&statsMux);
}

static void addMatched() {
    portENTER_CRITICAL(&statsMux);
    matched++;
    portEXIT_CRITICAL(&statsMux);
}

static void getStats(StatsBody &s) {
    portENTER_CRITICAL(&statsMux);

    s = {
        matched,
        unmatched,
        orphan,
        drops
    };

    portEXIT_CRITICAL(&statsMux);
}


// -------------------- Timestamp matching --------------------

static bool nearTs(uint32_t a, uint32_t b) {
    int32_t d = (int32_t)(a - b);

    return (
        d >= -TS_TOL_US &&
        d <= TS_TOL_US
    );
}


// -------------------- Promiscuous callback --------------------

static void promiscCallback(
    void *buffer,
    wifi_promiscuous_pkt_type_t type
) {
    if (
        type != WIFI_PKT_MGMT &&
        type != WIFI_PKT_DATA
    ) {
        return;
    }

    const wifi_promiscuous_pkt_t *p =
        (const wifi_promiscuous_pkt_t *)buffer;

    uint32_t n = p->rx_ctrl.sig_len;

    if (n > 160) {
        n = 160;
    }

    const uint8_t *d = p->payload;

    for (
        uint32_t i = 24;
        i + sizeof(TxPayload) <= n;
        ++i
    ) {
        if (memcmp(d + i, SURA_MAGIC, 4) == 0) {
            TxPayload t;

            memcpy(
                &t,
                d + i,
                sizeof(t)
            );

            PromEv ev = {
                p->rx_ctrl.timestamp,
                t.boot_id,
                t.seq,
                t.tx_us
            };

            if (xQueueSend(promQ, &ev, 0) != pdTRUE) {
                addDrop();
            }

            return;
        }
    }
}


// -------------------- CSI callback --------------------

static void csiCallback(
    void *ctx,
    wifi_csi_info_t *info
) {
    (void)ctx;

    if (
        !info ||
        !info->buf ||
        info->len <= 0 ||
        info->len > CSI_MAX
    ) {
        return;
    }

    CsiEv ev;

    ev.ts = info->rx_ctrl.timestamp;
    ev.rssi = info->rx_ctrl.rssi;
    ev.len = info->len;

    ev.flags =
        (info->rx_ctrl.sig_mode & 0x03) |
        (info->rx_ctrl.cwb ? FL_CWB40 : 0) |
        (info->rx_ctrl.stbc ? FL_STBC : 0);

#if CONFIG_IDF_TARGET_ESP32S3
    if (info->first_word_invalid) {
        ev.flags |= FL_FIRST_WORD_INVALID;
    }
#endif

    memcpy(
        ev.buf,
        info->buf,
        info->len
    );

    if (xQueueSend(csiQ, &ev, 0) != pdTRUE) {
        addDrop();
    }
}


// -------------------- Output frame creation --------------------

static bool queueFrame(
    uint8_t type,
    uint8_t flags,
    int8_t rssi,
    uint32_t boot,
    uint32_t seq,
    uint32_t txus,
    uint32_t rxts,
    const uint8_t *body,
    uint16_t len
) {
    OutFrame f;

    size_t total =
        sizeof(FrameHdr) +
        len +
        2;

    if (total > sizeof(f.bytes)) {
        addDrop();
        return false;
    }

    FrameHdr h = {
        0xA5,
        0x5A,
        type,
        RX_ID,
        len,
        flags,
        rssi,
        boot,
        seq,
        txus,
        rxts
    };

    memcpy(
        f.bytes,
        &h,
        sizeof(h)
    );

    if (len && body) {
        memcpy(
            f.bytes + sizeof(h),
            body,
            len
        );
    }

    uint16_t crc = sura_crc16(
        f.bytes,
        sizeof(h) + len
    );

    memcpy(
        f.bytes + sizeof(h) + len,
        &crc,
        2
    );

    f.len = (uint16_t)total;

    if (xQueueSend(outQ, &f, 0) != pdTRUE) {
        addDrop();
        return false;
    }

    return true;
}


static bool queueCsi(
    const CsiEv &c,
    bool haveProm,
    const PromEv &p
) {
    uint8_t flags =
        c.flags |
        (haveProm ? 0 : FL_UNMATCHED);

    return queueFrame(
        FT_CSI,
        flags,
        c.rssi,
        haveProm ? p.boot : 0,
        haveProm ? p.seq : 0xFFFFFFFF,
        haveProm ? p.txus : 0,
        c.ts,
        (const uint8_t *)c.buf,
        c.len
    );
}


// -------------------- USB writer task --------------------

// Only this task writes to Serial.
// If host reading pauses, outQ can temporarily absorb the delay.
// Once it fills, frame drops are counted rather than blocking CSI pairing.

static void usbWriterTask(void *) {
    OutFrame f;

    for (;;) {
        if (
            xQueueReceive(
                outQ,
                &f,
                portMAX_DELAY
            ) != pdTRUE
        ) {
            continue;
        }

        size_t sent = 0;

        while (sent < f.len) {
            int avail = Serial.availableForWrite();

            if (avail <= 0) {
                vTaskDelay(1);
                continue;
            }

            size_t chunk = (size_t)avail;

            if (chunk > f.len - sent) {
                chunk = f.len - sent;
            }

            size_t n = Serial.write(
                f.bytes + sent,
                chunk
            );

            if (n) {
                sent += n;
            } else {
                vTaskDelay(1);
            }
        }
    }
}


// -------------------- CSI / metadata pairing task --------------------

static void pairingTask(void *) {
    static PromEv pp[PROM_PENDING];
    static uint32_t pt[PROM_PENDING];
    static bool pu[PROM_PENDING] = {};

    static CsiEv cp[CSI_PENDING];
    static uint32_t ct[CSI_PENDING];
    static bool cu[CSI_PENDING] = {};

    PromEv p;
    CsiEv c;

    uint32_t lastStats = millis();

    for (;;) {
        bool worked = false;

        // Process TX metadata events.
        while (xQueueReceive(promQ, &p, 0) == pdTRUE) {
            worked = true;

            bool found = false;

            for (
                int i = 0;
                i < CSI_PENDING && !found;
                ++i
            ) {
                if (
                    cu[i] &&
                    nearTs(cp[i].ts, p.ts)
                ) {
                    if (queueCsi(cp[i], true, p)) {
                        addMatched();
                    }

                    cu[i] = false;
                    found = true;
                }
            }

            if (!found) {
                int slot = -1;

                for (int i = 0; i < PROM_PENDING; ++i) {
                    if (!pu[i]) {
                        slot = i;
                        break;
                    }
                }

                if (slot < 0) {
                    uint32_t now = millis();
                    uint32_t oldestAge = 0;

                    slot = 0;

                    for (int i = 0; i < PROM_PENDING; ++i) {
                        uint32_t age = now - pt[i];

                        if (age > oldestAge) {
                            oldestAge = age;
                            slot = i;
                        }
                    }

                    addOrphan();
                    addDrop();
                }

                pp[slot] = p;
                pt[slot] = millis();
                pu[slot] = true;
            }
        }


        // Process CSI events.
        while (xQueueReceive(csiQ, &c, 0) == pdTRUE) {
            worked = true;

            bool found = false;

            for (
                int i = 0;
                i < PROM_PENDING && !found;
                ++i
            ) {
                if (
                    pu[i] &&
                    nearTs(pp[i].ts, c.ts)
                ) {
                    if (queueCsi(c, true, pp[i])) {
                        addMatched();
                    }

                    pu[i] = false;
                    found = true;
                }
            }

            if (!found) {
                int slot = -1;

                for (int i = 0; i < CSI_PENDING; ++i) {
                    if (!cu[i]) {
                        slot = i;
                        break;
                    }
                }

                if (slot < 0) {
                    uint32_t now = millis();
                    uint32_t oldestAge = 0;

                    slot = 0;

                    for (int i = 0; i < CSI_PENDING; ++i) {
                        uint32_t age = now - ct[i];

                        if (age > oldestAge) {
                            oldestAge = age;
                            slot = i;
                        }
                    }

                    addUnmatched();
                    addDrop();
                }

                cp[slot] = c;
                ct[slot] = millis();
                cu[slot] = true;
            }
        }


        // Expire CSI events that did not find metadata.
        uint32_t now = millis();

        for (int i = 0; i < CSI_PENDING; ++i) {
            if (
                cu[i] &&
                (uint32_t)(now - ct[i]) > WAIT_MS
            ) {
                cu[i] = false;
                addUnmatched();
            }
        }


        // Expire metadata events that did not find CSI.
        for (int i = 0; i < PROM_PENDING; ++i) {
            if (
                pu[i] &&
                (uint32_t)(now - pt[i]) > WAIT_MS
            ) {
                pu[i] = false;
                addOrphan();
            }
        }


        // Send statistics once per second.
        if ((uint32_t)(now - lastStats) >= 1000) {
            lastStats = now;

            StatsBody s;
            getStats(s);

            queueFrame(
                FT_STATS,
                0,
                0,
                0,
                0,
                0,
                0,
                (const uint8_t *)&s,
                sizeof(s)
            );
        }

        if (!worked) {
            vTaskDelay(1);
        }
    }
}


// -------------------- Setup --------------------

static void fatal(
    const char *where,
    esp_err_t err
) {
    Serial.printf(
        "RX%d_FATAL %s err=%d\n",
        RX_ID,
        where,
        (int)err
    );

    while (true) {
        delay(1000);
    }
}

void setup() {
    Serial.setTxBufferSize(32768);
    Serial.begin(115200);

    delay(500);

    promQ = xQueueCreate(
        PROM_Q_LEN,
        sizeof(PromEv)
    );

    csiQ = xQueueCreate(
        CSI_Q_LEN,
        sizeof(CsiEv)
    );

    outQ = xQueueCreate(
        OUT_Q_LEN,
        sizeof(OutFrame)
    );

    if (!promQ || !csiQ || !outQ) {
        Serial.printf(
            "RX%d_FATAL queue allocation failed free_heap=%u\n",
            RX_ID,
            (unsigned)ESP.getFreeHeap()
        );

        while (true) {
            delay(1000);
        }
    }

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_err_t e;

    e = esp_wifi_set_bandwidth(
        WIFI_IF_STA,
        WIFI_BW_HT20
    );

    if (e != ESP_OK) {
        fatal("set_bandwidth", e);
    }

    wifi_promiscuous_filter_t filter = {};

    filter.filter_mask =
        WIFI_PROMIS_FILTER_MASK_MGMT |
        WIFI_PROMIS_FILTER_MASK_DATA;

    e = esp_wifi_set_promiscuous_filter(&filter);

    if (e != ESP_OK) {
        fatal("set_promiscuous_filter", e);
    }

    e = esp_wifi_set_promiscuous_rx_cb(
        &promiscCallback
    );

    if (e != ESP_OK) {
        fatal("set_promiscuous_callback", e);
    }

    e = esp_wifi_set_channel(
        WIFI_CH,
        WIFI_SECOND_CHAN_NONE
    );

    if (e != ESP_OK) {
        fatal("set_channel", e);
    }

    wifi_csi_config_t cfg = {};

    cfg.lltf_en = true;
    cfg.htltf_en = true;
    cfg.stbc_htltf2_en = true;
    cfg.ltf_merge_en = true;
    cfg.channel_filter_en = false;
    cfg.manu_scale = false;

    e = esp_wifi_set_csi_config(&cfg);

    if (e != ESP_OK) {
        fatal("set_csi_config", e);
    }

    e = esp_wifi_set_csi_rx_cb(
        &csiCallback,
        nullptr
    );

    if (e != ESP_OK) {
        fatal("set_csi_callback", e);
    }

    e = esp_wifi_set_csi(true);

    if (e != ESP_OK) {
        fatal("enable_csi", e);
    }

    e = esp_wifi_set_promiscuous(true);

    if (e != ESP_OK) {
        fatal("enable_promiscuous", e);
    }

    if (
        xTaskCreatePinnedToCore(
            usbWriterTask,
            "sura_usb",
            4096,
            nullptr,
            4,
            nullptr,
            0
        ) != pdPASS ||

        xTaskCreatePinnedToCore(
            pairingTask,
            "sura_pair",
            8192,
            nullptr,
            5,
            nullptr,
            1
        ) != pdPASS
    ) {
        Serial.printf(
            "RX%d_FATAL task creation failed\n",
            RX_ID
        );

        while (true) {
            delay(1000);
        }
    }

    Serial.printf(
        "RX_READY id=%d ch=%d heap_after_setup=%u\n",
        RX_ID,
        WIFI_CH,
        (unsigned)ESP.getFreeHeap()
    );
}

void loop() {
    delay(1000);
}

// SURA TX: numbered ESP-NOW broadcast.
// Sequence counts every send attempt.
// api_ok/api_fail are immediate API results.
// mac_ok/mac_fail are asynchronous MAC results.
// A successful broadcast MAC result is NOT an acknowledgement from an RX board.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <esp_timer.h>
#include <esp_idf_version.h>
#include "sura_proto.h"

#ifndef WIFI_CH
#define WIFI_CH 1
#endif

static const uint8_t BCAST[6] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

static TxPayload payload;

static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

static volatile uint32_t apiOk = 0;
static volatile uint32_t apiFail = 0;
static volatile uint32_t macOk = 0;
static volatile uint32_t macFail = 0;

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 5, 0)

static void sendCallback(
    const wifi_tx_info_t *info,
    esp_now_send_status_t status
) {
    (void)info;

#else

static void sendCallback(
    const uint8_t *mac,
    esp_now_send_status_t status
) {
    (void)mac;

#endif

    portENTER_CRITICAL(&mux);

    if (status == ESP_NOW_SEND_SUCCESS) {
        macOk++;
    } else {
        macFail++;
    }

    portEXIT_CRITICAL(&mux);
}

static void fatal(const char *where, esp_err_t err) {
    Serial.printf(
        "TX_FATAL %s err=%d\n",
        where,
        (int)err
    );

    while (true) {
        delay(1000);
    }
}

void setup() {
    Serial.begin(115200);
    delay(700);

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

    e = esp_wifi_set_channel(
        WIFI_CH,
        WIFI_SECOND_CHAN_NONE
    );

    if (e != ESP_OK) {
        fatal("set_channel", e);
    }

    e = esp_now_init();

    if (e != ESP_OK) {
        fatal("esp_now_init", e);
    }

    esp_now_peer_info_t peer = {};

    memcpy(peer.peer_addr, BCAST, 6);

    peer.channel = WIFI_CH;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;

    e = esp_now_add_peer(&peer);

    if (e != ESP_OK) {
        fatal("add_peer", e);
    }

#if ESP_ARDUINO_VERSION_MAJOR >= 3

    esp_now_rate_config_t rate = {};

    rate.phymode = WIFI_PHY_MODE_HT20;
    rate.rate = WIFI_PHY_RATE_MCS0_LGI;
    rate.ersu = false;
    rate.dcm = false;

    e = esp_now_set_peer_rate_config(
        BCAST,
        &rate
    );

#else

    e = esp_wifi_config_espnow_rate(
        WIFI_IF_STA,
        WIFI_PHY_RATE_MCS0_LGI
    );

#endif

    if (e != ESP_OK) {
        fatal("set_rate", e);
    }

    e = esp_now_register_send_cb(sendCallback);

    if (e != ESP_OK) {
        fatal("register_send_cb", e);
    }

    memcpy(payload.magic, SURA_MAGIC, 4);

    payload.boot_id = esp_random();
    payload.seq = 0;
    payload.tx_us = 0;

    Serial.printf(
        "TX_READY ch=%d rate=%dHz boot_id=%08x\n",
        WIFI_CH,
        TX_HZ,
        payload.boot_id
    );

    Serial.println(
        "Broadcast MAC success does not prove reception by any RX."
    );
}

void loop() {
    static TickType_t lastWake = xTaskGetTickCount();
    static uint32_t lastPrint = 0;

    vTaskDelayUntil(
        &lastWake,
        pdMS_TO_TICKS(1000 / TX_HZ)
    );

    payload.tx_us = (uint32_t)esp_timer_get_time();

    esp_err_t result = esp_now_send(
        BCAST,
        reinterpret_cast<const uint8_t *>(&payload),
        sizeof(payload)
    );

    portENTER_CRITICAL(&mux);

    if (result == ESP_OK) {
        apiOk++;
    } else {
        apiFail++;
    }

    portEXIT_CRITICAL(&mux);

    // Sequence counts attempts, including immediate API failures.
    payload.seq++;

    if (millis() - lastPrint >= 2000) {
        lastPrint = millis();

        uint32_t a, f, s, m;

        portENTER_CRITICAL(&mux);

        a = apiOk;
        f = apiFail;
        s = macOk;
        m = macFail;

        portEXIT_CRITICAL(&mux);

        Serial.printf(
            "seq=%u api_ok=%u api_fail=%u mac_ok=%u mac_fail=%u pending~=%u\n",
            payload.seq,
            a,
            f,
            s,
            m,
            (uint32_t)(a - s - m)
        );
    }
}

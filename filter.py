#include "securec.h"
#include "spo2_common.h"
#include "spo2_filtfilt.h"
#define SPO2_ALG_VERSION   0x01

#define RAD_TO_DEG  (57.2958f)

#define PPG_SAMPLE_RATE    (4)
#define PPG_CALC_FREQ      (25)
#define PPG_MEAN_LEN     (6)
#define PPG_DC_MAX         (100000000)
#define PPG_DC_MIN         (1000)
#define PPG_FILT_LEN    (PPG_CALC_FREQ * PPG_MEAN_LEN)
#define PPG_FILT_MOV_LENGTH  (2 * PPG_CALC_FREQ)
#define PPG_FILT_OVERLAP_LENGTH     (PPG_FILT_LEN -PPG_FILT_MOV_LENGTH)
#define PPG_MAX_FEAT_LEN (PPG_FILT_LEN / 6)  // <240bpm
#define PPG_FLITFLIT_ORDER  (6)
#define PPG_FLITFLIT_EXTRALEN (3 * PPG_FLITFLIT_ORDER)
#define PPG_FLITFLIT_LEN  (PPG_FILT_LEN + 2 * PPG_FLITFLIT_EXTRALEN)

#define PPG_OVERLAP_LENGTH (PPG_CALC_FREQ * 3)
#define PPG_FFT_LENGTH     (128)
#define PPG_FFTBUFFER_LEN (PPG_FFT_LENGTH * 2 + 1)
#define SPO2_MEM_BUFF_SIZE (1500)

#ifndef LOG_FOR_SPO2ALG
#define LOG_FOR_SPO2ALG       18              /* log for algorithm module */
#endif // !LOG_FOR_ALG

#ifndef MIN
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#endif

/* Bandpass filter 0.5-3Hz,Fs=25Hz*/
static float b[PPG_FLITFLIT_ORDER + 1] = { 0.0180989330075144f, 0.0f, -0.0542967990225433f, 0.0f, 0.0542967990225433f, 0.0f, -0.0180989330075144f };
static float a[PPG_FLITFLIT_ORDER + 1] = { 1.0, -4.5286632186525786f, 8.7323585632218474f, -9.2346678390704664f, 5.6724234164823990f, -1.9190260509732218f, 0.2780599176345465f };
static float zi[PPG_FLITFLIT_ORDER] = { -0.0180989330075518f, -0.0180989330073827f, 0.0361978660148346f, 0.0361978660151794f, -0.0180989330075757f, -0.0180989330075041f };

/* 数据翻转模块 */
static void dataFlip(float* dataBuffer, uint16_t dataBufferLen)
{
    if (dataBuffer == NULL) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:dataFlip() input null %d\r\n", 0);
        return;
    }

    if (dataBufferLen == 0) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:dataFlip() input length error %d\r\n", 0);
        return;
    }

    float temp;
    for (uint16_t i = 0; i < dataBufferLen / 2; i++) {
        temp = dataBuffer[i];
        dataBuffer[i] = dataBuffer[dataBufferLen - 1 - i];
        dataBuffer[dataBufferLen - 1 - i] = temp;
    }
}

/* IIR滤波模块 */
static void iirFilt(float* dataBuffer, uint16_t ffiltBuffLen, const float* ziBuff, uint8_t filtOrder)
{
    if (dataBuffer == NULL || ziBuff == NULL) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:iirFilt() input null %d\r\n", 0);
        return;
    }

    if (ffiltBuffLen == 0) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:iirFilt() input length error %d\r\n", 0);
        return;
    }

    uint16_t i, j;
    float filtTempBuff[PPG_FLITFLIT_ORDER + 1] = { 0 };

    for (i = 0; i < filtOrder; i++) {
        filtTempBuff[i + 1] = ziBuff[i];
    }

    for (j = 0; j < ffiltBuffLen; j++) {
        for (i = 0; i < filtOrder; i++) {
            filtTempBuff[i] = filtTempBuff[i + 1];
        }

        filtTempBuff[filtOrder] = 0;

        for (i = 0; i < (filtOrder + 1); i++) {
            filtTempBuff[i] += dataBuffer[j] * b[i];
        }

        for (i = 0; i < filtOrder; i++) {
            filtTempBuff[i + 1] -= filtTempBuff[0] * a[i + 1];
        }

        dataBuffer[j] = filtTempBuff[0];
    }
}

/* 双向滤波结果获取模块 */
static void getFfiltBuff(float* ffiltBuff, const float* inBuff, uint16_t ffiltBuffLen, uint16_t inBuffLen, uint16_t extraLen)
{
    if (ffiltBuff == NULL || inBuff == NULL) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:getFfiltBuff() input null %d\r\n", 0);
        return;
    }

    if (inBuffLen == 0 || extraLen == 0 || ffiltBuffLen < inBuffLen + 2 * extraLen) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:getFfiltBuff() input length error %d\r\n", 0);
        return;
    }

    uint16_t i;
    float headVar;
    float endVar;

    headVar = 2 * inBuff[0];
    endVar = 2 * inBuff[(inBuffLen - 1)];

    for (i = 0; i < extraLen; i++) {
        ffiltBuff[i] = headVar - inBuff[extraLen - i];
    }

    for (i = 0; i < inBuffLen; i++) {
        ffiltBuff[i + extraLen] = inBuff[i];
    }

    for (i = 0; i < extraLen; i++) {
        ffiltBuff[i + (inBuffLen + extraLen)] = endVar - inBuff[(inBuffLen - 2) - i];
    }
}

/* 双向滤波计算模块 */
static void ffiltProcess(float* ffiltBuff, uint16_t filtLen)
{
    if (ffiltBuff == NULL) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:ffiltProcess() input null %d\r\n", 0);
        return;
    }

    if (filtLen < PPG_FLITFLIT_LEN) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:ffiltProcess() input length error %d\r\n", 0);
        return;
    }

    float ziBuff[PPG_FLITFLIT_ORDER] = { 0 };

#ifdef PC_TEST
    for (uint16_t i = 0; i < PPG_FLITFLIT_ORDER; i++) {
        ziBuff[i] = zi[i] * ffiltBuff[0];
    }
#else
    float coefTemp = ffiltBuff[0];
    arm_scale_f32(&zi[0], coefTemp, &ziBuff[0], PPG_FLITFLIT_ORDER);
#endif

    iirFilt(ffiltBuff, PPG_FLITFLIT_LEN, ziBuff, PPG_FLITFLIT_ORDER);
    dataFlip(ffiltBuff, PPG_FLITFLIT_LEN);

#ifdef PC_TEST
    for (uint16_t i = 0; i < PPG_FLITFLIT_ORDER; i++) {
        ziBuff[i] = zi[i] * ffiltBuff[0];
    }
#else
    coefTemp = ffiltBuff[0];
    arm_scale_f32(&zi[0], coefTemp, &ziBuff[0], PPG_FLITFLIT_ORDER);
#endif

    iirFilt(ffiltBuff, PPG_FLITFLIT_LEN, ziBuff, PPG_FLITFLIT_ORDER);
    dataFlip(ffiltBuff, PPG_FLITFLIT_LEN);
}

/* 双向滤波主接口 */
void spo2FiltFilt(float* dataBuffer, uint16_t buffLen)
{
    if (dataBuffer == NULL) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:spo2FiltFilt() input null %d\r\n", 0);
        return;
    }

    if (buffLen < PPG_FLITFLIT_LEN) {
        Spo2CalLogPrintf(LOG_FOR_SPO2ALG, LOG_ERROR_SPO2, "spo2Alg:spo2FiltFilt() buffLen error %d\r\n", 0);
        return;
    }

    float ffiltBuff[PPG_FLITFLIT_LEN] = { 0 };

    getFfiltBuff(ffiltBuff, dataBuffer, PPG_FLITFLIT_LEN, PPG_FILT_LEN, PPG_FLITFLIT_EXTRALEN);

    ffiltProcess(ffiltBuff, PPG_FLITFLIT_LEN);

    for (uint16_t i = 0; i < PPG_FILT_LEN; i++) {
        dataBuffer[i] = ffiltBuff[PPG_FLITFLIT_EXTRALEN + i];
    }
}

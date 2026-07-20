#!/bin/sh

set -eu

firebase_source="${PROJECT_DIR}/Runner/GoogleService-Info.plist"
firebase_destination="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/GoogleService-Info.plist"

if [ -f "${firebase_source}" ]; then
  mkdir -p "$(dirname "${firebase_destination}")"
  cp "${firebase_source}" "${firebase_destination}"
fi

case "${CONFIGURATION}" in
  Release|Profile)
    problems=""
    test_app_id="ca-app-pub-3940256099942544~1458002511"
    test_banner_id="ca-app-pub-3940256099942544/2435281174"
    test_native_id="ca-app-pub-3940256099942544/3986624511"
    test_rewarded_id="ca-app-pub-3940256099942544/1712485313"

    if [ ! -f "${firebase_source}" ]; then
      problems="${problems}\nRunner/GoogleService-Info.plist is missing."
    fi
    if [ -z "${KEIBA_ADMOB_APP_ID:-}" ] || [ "${KEIBA_ADMOB_APP_ID}" = "${test_app_id}" ]; then
      problems="${problems}\nKEIBA_ADMOB_APP_ID is missing or still uses Google's demo ID."
    fi
    if [ -z "${KEIBA_BANNER_AD_UNIT_ID:-}" ] || [ "${KEIBA_BANNER_AD_UNIT_ID}" = "${test_banner_id}" ]; then
      problems="${problems}\nKEIBA_BANNER_AD_UNIT_ID is missing or still uses Google's demo ID."
    fi
    if [ -z "${KEIBA_NATIVE_MORE_AD_UNIT_ID:-}" ] || [ "${KEIBA_NATIVE_MORE_AD_UNIT_ID}" = "${test_native_id}" ]; then
      problems="${problems}\nKEIBA_NATIVE_MORE_AD_UNIT_ID is missing or still uses Google's demo ID."
    fi
    if [ -z "${KEIBA_CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID:-}" ] || [ "${KEIBA_CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID}" = "${test_rewarded_id}" ]; then
      problems="${problems}\nKEIBA_CUSTOM_PREDICTION_REWARDED_AD_UNIT_ID is missing or still uses Google's demo ID."
    fi

    if [ -n "${problems}" ]; then
      printf "iOS release configuration is invalid:%b\n" "${problems}" >&2
      exit 1
    fi
    ;;
esac

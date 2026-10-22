# Бесплатная замена MuMu: проверка документации, 23 октября 2026

Ничего не устанавливалось, устройство MuMu и аккаунты не изменялись. Реальные матчи: 0/5.

| Кандидат | Цена/Apple Silicon | ADB/root и наш парсер | Clash Royale |
|---|---|---|---|
| BlueStacks Air | Официально полностью бесплатно, Apple Silicon; требования macOS11+, M1–M4, 8GB RAM/12GB диск | В прочитанной официальной документации Air штатные root/ADB не подтверждены. Инструкция ADB для BlueStacks5 не доказывает работу Air. ABI Android и доступ к /proc процесса следует проверить на устройстве | Работа конкретной текущей версии на этом Mac не проверена; общего обещания «Android games» недостаточно |
| Android Emulator, AOSP arm64-v8a | Есть официальная MacARM версия AndroidStudio/SDK и ARM64 образы, подписка эмулятора не требуется | Google прямо документирует adb root на AOSP; образы GooglePlayStore root не дают. Это наиболее обоснованный кандидат для memory parser, но доступ к /proc/PID/mem, SELinux и native binary нужно проверить отдельно | Не подтверждено: у AOSP нет GoogleApps/Services; нужны совместимый APK, запуск, сеть и вход. Успех нельзя обещать до проверки |
| MuMuPlayer for Mac | Официально 7дней только для ранее не использовавшего Mac, затем оплачиваемое членство | Существующий профиль/парсер ранее работал, сейчас запуск заблокирован истёкшей подпиской | В существующем профиле работал; постоянного бесплатного Mac режима официальная статья не предлагает |

Рекомендация для дальнейшего обратимого эксперимента: отдельный AOSP ARM64 AVD, сохранив MuMu. Сначала без матчей проверить запуск игры, adb uid0, ABI arm64-v8a, доступ к maps/mem, точную версию и SHA libg, загрузку нашего native parser, корректность side/hand/tick/координат. Затем заново проверить экранную калибровку 1440x2560. Возможность выполнить adb shell сама по себе недостаточна.

Из текущего кода: scripts/gamestate/arm64_entities.py требует уже-root ADB, arm64-v8a, конкретную версию игры/SHA libg; читает /proc/PID/maps и /proc/PID/mem. query_arm64.py использует /data/local/tmp/hasty_arm64_entities_20261022_v8. Наличие бинарника/прав в новом эмуляторе не предполагается автоматически. Упомянутые проверки — следующий этап, сейчас не выполнялись.

Официальные источники (проверены браузером):

- BlueStacks Air, FAQ «completely free», AppleSilicon: https://www.bluestacks.com/mac
- Требования Air: https://support.bluestacks.com/hc/en-us/articles/32272913555597-System-specifications-for-installing-BlueStacks-Air
- Меню Air (root/ADB в описании отсутствуют; это отсутствие подтверждения, не доказательство невозможности): https://support.bluestacks.com/hc/en-us/articles/32272893259533-How-to-use-the-Settings-Menu-on-BlueStacks-Air
- ADB статья относится к BlueStacks5: https://support.bluestacks.com/hc/en-us/articles/23925869130381-How-to-enable-Android-Debug-Bridge-on-BlueStacks-5
- AndroidStudio MacARM и SDK downloads: https://developer.android.com/studio
- AOSP root и запрет root у GooglePlay образов: https://developer.android.com/studio/run/managing-avds
- AndroidEmulator commandline/ADB: https://developer.android.com/studio/run/emulator-commandline
- Официальный исходник Google с ARM64/M1 поддержкой: https://android.googlesource.com/platform/tools/emulator/+/refs/tags/studio-2025.2.1-alpha01
- MuMuMac trial/членство: https://www.mumuplayer.com/help/mac/free-trial-and-membership.html

Сторонние root-патчеры и сайты, имитирующие официальный MuMu, не использовались. Установка/перенос/новые профили и оплата не выполнялись.

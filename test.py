from analyse import Analyse

DEVICE_ID = 1


def main() -> None:
    success = Analyse(DEVICE_ID)
    print(success)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()

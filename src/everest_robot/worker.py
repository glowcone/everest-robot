"""Worker process entry point."""

from everest_robot.workflow import QUEUE_NAME, app


def main() -> None:
    print(f"robot worker listening on queue {QUEUE_NAME}")
    try:
        app.start_worker()
    except KeyboardInterrupt:
        print("robot worker stopped")
    finally:
        app.close()


if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(
        description="Plot een kolom uit meerdere CSV bestanden"
    )

    parser.add_argument(
        "files",
        help="CSV bestanden gescheiden door komma's"
    )

    parser.add_argument(
        "column",
        help="Kolom die je wil plotten"
    )

    args = parser.parse_args()

    # CSV-bestanden opsplitsen
    files = [f.strip() for f in args.files.split(",")]

    plt.figure(figsize=(10, 5))

    for file in files:
        # CSV inlezen
        df = pd.read_csv(file)

        # Controleren of kolommen bestaan
        if "Step" not in df.columns:
            print(f"'Step' kolom ontbreekt in {file}")
            continue

        if args.column not in df.columns:
            print(f"Kolom '{args.column}' bestaat niet in {file}")
            print("Beschikbare kolommen:", list(df.columns))
            continue

        # Bestandsnaam zonder extensie gebruiken als label
        label = os.path.splitext(os.path.basename(file))[0]

        # Plotten
        plt.plot(
            df[args.column],
            label=label
        )

    c_label = "Average reward per timestep"
    # Grafiek afwerken
    plt.title(c_label)
    plt.xlabel("Step")
    plt.ylabel(c_label)
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # Opslaan in map van eerste CSV
    output_dir = os.path.dirname(os.path.abspath(files[0]))
    output_file = os.path.join(
        output_dir,
        f"{args.column.replace(' ', '_')}_comparison.png"
    )

    plt.savefig(output_file)

    print(f"Grafiek opgeslagen als: {output_file}")


if __name__ == "__main__":
    main()
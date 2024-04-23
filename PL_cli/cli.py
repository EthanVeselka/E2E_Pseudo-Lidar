import click
import sys

from PL_cli.edit_config import edit_config
from PL_cli.run_script import run_script

@click.command()
def main():
    click.echo("This is the CLI for the Pseudo-LiDAR project. Type 'help' at any time for a list of options.") 

    while True:
        command = click.prompt("\nEnter a command")

        if command == "exit":
            click.echo("Exiting...")
            break

        if command == "help":
            click.echo("")
            click.echo("edit: Edit a configuration file")
            click.echo("collect : Run the data collection script")
            click.echo("")
            click.echo("help: Display a helpful message")
            click.echo("exit: Close the CLI")
            continue

        if command == "edit":
            option = ""
            while option not in ["data collection", "sampling", "cancel"]:
                option = click.prompt("Which config file would you like to edit? (Data collection / Sampling / Cancel)", type=str).lower()
                if option == "cancel":
                    break

            if option == "help":
                click.echo("Options: data collection, sampling")
                continue
            if option == "data collection":
                file_path = "carla_data/config.ini"
            if option == "sampling":
                file_path = "processing/config.ini"

            while True:
                key = click.prompt("Enter the key to modify, or enter 'done' to stop")

                if key == "done":
                    break

                file_path = ""
                if option == "data collection":
                    if key not in ["help", "data_path", "ego_behavior", "external_behavior", "weather", "map", "carla_python_path", "poll_rate", "camera_x", "camera_y", "camera_fov"]:
                        click.echo("Invalid key.")
                        continue
                    if key == "help":
                        click.echo("Options: data_path, ego_behavior, external_behavior, weather, map, carla_python_path, poll_rate, camera_x, camera_y, camera_fov")
                        continue
                    file_path = "carla_data/config.ini"
                elif option == "sampling":
                    if key not in ["help", "data_path", "ego_behavior", "external_behavior", "weather", "map", "all", "splits", "sample_size"]:
                        click.echo("Invalid key.")
                        continue
                    if key == "help":
                        click.echo("Options: data_path, ego_behavior, external_behavior, weather, map, all, splits, sample_size")
                        continue
                    file_path = "processing/config.ini"
                

                value = click.prompt("Enter the new value for the key")
                if value == "help":
                    click.echo("Check the README for valid values for each key: https://github.com/EthanVeselka/E2E_Pseudo-Lidar")
                    continue

                try:
                    edit_config(key, value, file_path)
                except Exception as e:
                    click.echo(f"Error: {e}")
                    click.echo("")

            continue

        if command == "collect":
            click.echo("Do you want to run the data collection script? (y/n)")
            response = click.prompt("")
            if response in ["y", "yes", "Y", "Yes"]:
                click.echo("Running carla_data/carla_client.py ...")
                        
                # run data collection script
                try:
                    run_script("carla_data\carla_client.py")
                except Exception as e:
                    click.echo(f"Error: {e}")

                break
            else:
                continue

        if command == "view":
            click.echo("Do you want to run the data viewer? (y/n)")
            response = click.prompt("")
            if response in ["y", "yes", "Y", "Yes"]:
                click.echo("Running carla_data/data_viewer.py ...")
                        
                # run data viewer script
                try:
                    run_script("carla_data\data_viewer.py")
                except Exception as e:
                    click.echo(f"Error: {e}")

                break
            else:
                continue

if __name__ == "__main__":
    main()

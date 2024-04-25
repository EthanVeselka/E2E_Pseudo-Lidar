import click

from PL_cli.edit_config import edit_config
from PL_cli.run_script import run_script

@click.command()
def main():
    click.echo("This is the CLI for the Pseudo-LiDAR project. Type 'help' at any time for a list of options.") 

    while True:
        command = click.prompt("\nEnter a command")
        command = command.lower()
        command = command.strip()

        if command == "exit":
            click.echo("Exiting...")
            break

        if command == "help":
            click.echo("\n".join([
                "", 
                "edit: Edit a configuration file", 
                "collect: Run the data collection script", 
                "view: Run the data viewer", 
                "", 
                "help: Display a helpful message", 
                "exit: Close the CLI"
                ]))
            continue

        if command == "edit":
            option = ""
            
            while True:
                option = click.prompt("Which config file would you like to edit? (Data collection / Sampling / Cancel)", type=str).lower()
                option = option.lower()

                if option in ["cancel", "done", "quit", "exit"]:
                    break
                elif option == "help":
                    click.echo("Options: Data collection (carla_data/config.ini), Sampling (processing/config.ini)")
                    continue
                elif option in ["data collection", "sampling"]:
                    while True:
                        file_path = ""
                        if option == "data collection":
                            # output contents of config file
                            with open("carla_data/config.ini", "r") as file:
                                click.echo()
                                click.echo(file.read())

                            key = click.prompt("Enter the key to modify, or enter 'done' to stop")

                            if key == "done":
                                break
                            if key not in ["help", "data_path", "ego_behavior", "external_behavior", "weather", "map", "carla_python_path", "poll_rate", "camera_x", "camera_y", "camera_fov"]:
                                click.echo("Invalid key.")
                                continue
                            if key == "help":
                                click.echo("Options: data_path, ego_behavior, external_behavior, weather, map, carla_python_path, poll_rate, camera_x, camera_y, camera_fov")
                                continue
                            file_path = "carla_data/config.ini"

                        elif option == "sampling":
                            # output contents of config file
                            with open("processing/config.ini", "r") as file:
                                click.echo()
                                click.echo(file.read())
                                
                            key = click.prompt("Enter the key to modify, or enter 'done' to stop")

                            if key == "done":
                                break

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
                    break
                
                else:
                    click.echo("Unknown option. Type 'help' for a list of options.")
                    continue

        if command == "collect":
            response = click.prompt("Do you want to run the data collection script? (y/n)")
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
            response = click.prompt("Do you want to run the data viewer? (y/n)")
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

        if command == "process":
            response = click.prompt("Do you want to run the processing script? (y/n)")
            if response in ["y", "yes", "Y", "Yes"]:
                click.echo("Running processing/scripts/process.sh ...")
                
                # run data processing script
                try:
                    run_script("processing/scripts/process.sh")
                except Exception as e:
                    click.echo(f"Error: {e}")

                break
            else:
                continue

        click.echo("Unknown command. Type 'help' for a list of options.")
        
if __name__ == "__main__":
    main()

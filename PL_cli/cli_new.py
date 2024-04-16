import click
import sys

from edit_config import edit_config


@click.command()
def main():
    click.echo("Welcome to your CLI!")

    while True:
        command = click.prompt("Enter a command (type 'exit' to quit):")

        if command == 'exit':
            click.echo("Exiting...")
            break

        if command == 'edit':
            try :
                edit()  
            except Exception as e:
                click.echo(f"Error: {e}")
                click.echo("")

@click.command()
@click.option("--key", type=str, help="The key to modify. Options: data_path, ego_behavior, external_behavior, weather, map, all, splits, sample_size, carla_python_path, poll_rate, camera_x, camera_y, camera_fov")
@click.option("--value", type=str, help="The new value for the key.")
@click.option("--file-path", type=str, help="Path to the config file.")

def edit(key, value, file_path):
    try:
        edit_config(key, value, file_path)
        click.echo("Configuration updated successfully.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}")
        click.echo("") #TODO: Add usage instructions
        sys.exit(1)

if __name__ == "__main__":
    main()

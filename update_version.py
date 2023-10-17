import semver
import toml

# Load the pyproject.toml file
with open("pyproject.toml", "r") as file:
    pyproject_data = toml.load(file)

# Get the current version from pyproject.toml
current_version = pyproject_data["tool"]["poetry"]["version"]

# Determine the new version based on the branch name or other logic
# You can customize this logic according to your needs
# In this example, we'll increment the PATCH version for any branch
new_version = semver.bump_patch(current_version)

# Update the version in pyproject.toml
pyproject_data["tool"]["poetry"]["version"] = new_version

# Write the updated data back to pyproject.toml
with open("pyproject.toml", "w") as file:
    toml.dump(pyproject_data, file)

# Print the updated version for reference
print(f"Updated version to {new_version}")

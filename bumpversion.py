import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release_type", help="Which version to bump (major, minor, revision)")
    args = parser.parse_args()

    with open("INDUSAnalysis/_version.py", "r") as vh:
        version_def = vh.read()
        version = version_def.split('=')[1].strip()[1:-1]

    version_list = [int(v) for v in version.strip().split('.')]

    if args.release_type == "major":
        version_list[0] += 1
        version_list[1] = 0
        version_list[2] = 0
    elif args.release_type == "minor":
        version_list[1] += 1
        version_list[2] = 0
    elif args.release_type == "revision":
        version_list[2] += 1
    else:
        raise ValueError("Undefined release type.")

    new_version = ".".join([str(v) for v in version_list])
    print("Updating version from %s to %s" % (version, new_version))

    with open("INDUSAnalysis/_version.py", "w") as vh:
        vh.write("version = '{}'".format(new_version))


if __name__ == "__main__":
    main()

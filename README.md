# x-img-diff

## Usage

```text
  ximgdiff [actual] [expected] [out] {OPTIONS}

    Compare two images

  OPTIONS:

      -h, --help                        Display this help menu.
      -v, --verbose                     Display debug logging messages.
      actual                            Actual image path
      expected                          Expected image path
      out                               Output image path
      "--" can be used to terminate flag options and force all following
      arguments to be treated as positional options
```

## Install
### Requirements

- OpenCV
- cmake

### Build

```sh
mkdir build; cd build
cmake ..
make
```

## License
MIT. See LICENSE file.

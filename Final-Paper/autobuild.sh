
CHECKSUM=""

make clean

while [[ 1 == 1 ]]; do
    CHECKSUM_NEW=$(sha256sum main.tex)

    if [[ "$CHECKSUM_NEW" != "$CHECKSUM" ]]; then
        make main.pdf
        CHECKSUM=$(sha256sum main.tex)
    fi

done
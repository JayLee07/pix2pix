
FILE=$facades
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
TAR_FILE=/data/jehyuk/imgdata/datasets/facades.tar.gz
TARGET_DIR=/data/jehyuk/imgdata/datasets/facades/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C /data/jehyuk/imgdata/datasets/
rm $TAR_FILE

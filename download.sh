wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz
tar zxvf LKH-3.0.6.tgz
cd LKH-3.0.6
make
mv LKH ../marvin/utils/
cd ..
rm LKH-3.0.6.tgz
rm -r LKH-3.0.6
mkdir marvin/data
mkdir marvin/data/train_graphs
mkdir marvin/data/test_graphs
mkdir marvin/data/val_graphs

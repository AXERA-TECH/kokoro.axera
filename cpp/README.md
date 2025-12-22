# CPP

## 交叉编译

在 PC 上完成（已在Ubuntu22.04上测试）

安装开发环境:
```
sudo apt update
sudo apt install build-essential cmake
```

获取交叉编译工具链: [地址](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz)  
将交叉编译工具链路径添加到PATH  

编译  
```
cd cpp
./download_bsp.sh
./build.sh
```

编译完成的bin在install目录下

其中kokoro是命令行demo，kokoro_srv是服务端
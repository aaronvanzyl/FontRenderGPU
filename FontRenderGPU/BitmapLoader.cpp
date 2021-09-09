#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void memory_test(vector<int>& arr) {
    arr.resize(10);
    arr[7] = 2000;


    char* test_bytes = new char[4]{ 1,2,3,4 };

    //memcpy(&arr[0], test_bytes, 3);
    copy(test_bytes, test_bytes + 3, arr.begin());

    cout << "test:";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << "\n";
}


static void load_font_bitmaps(string font_file, int width, int height, vector<char>& bitmaps, vector<int>& unicode_val, int& num_chars) {
    string numline;
    string bytesline;
    ifstream myfile(font_file);
    int size = width * height;
    if (myfile.is_open())
    {
        char* int_bytes = new char[4];

        myfile.read(int_bytes, 4);
        std::memcpy(&num_chars, int_bytes, 4);

        bitmaps.resize(size * num_chars);

        cout << num_chars << "\n";

        for (int char_idx = 0; char_idx < num_chars; char_idx++) {
            int unicode_val;
            myfile.read(int_bytes, 4);
            std::memcpy(&unicode_val, int_bytes, 4);

            cout << unicode_val << "\n";

            myfile.read(&bitmaps[size * char_idx], size);
        }
        myfile.close();
    }

    else cout << "Unable to open file";
}

static void render(vector<char>& bitmaps, int width, int height, int index) {
    int offset = width * height * index;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = int(bitmaps[offset + y * width + x]);
            cout << (val ? "0" : ".");
        }
        cout << "\n";
    }
    cout << "\n";
}
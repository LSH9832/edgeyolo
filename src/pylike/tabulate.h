#pragma once
#ifndef PYTHONLIKE_TABULATE_H
#define PYTHONLIKE_TABULATE_H


#include "./str.h"
#include <unordered_map>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>


#ifdef SAFE_DEFINE
#define TableSetStyle tabulate::tabulate.setDefaultStyle
#define TableSetAlign tabulate::tabulate.setAlign
#define TableClear tabulate::tabulate.clear()
#define TableHead tabulate::tabulate.setHead()
#define TableLine tabulate::tabulate.setContent()
#define TableCreate tabulate::tabulate
#define TableEndLine tabulate::tabulate.endLine()
#define TableShow tabulate::tabulate.toString
#define TSTYLE_PLAIN tabulate::TABLE_STYLE_PLAIN
#define TSTYLE_SIMPLE tabulate::TABLE_STYLE_SIMPLE
#define TSTYLE_GRID tabulate::TABLE_STYLE_GRID
#define TSTYLE_SIMPLE_GRID tabulate::TABLE_STYLE_SIMPLE_GRID
#define TSTYLE_ROUNDED_GRID tabulate::TABLE_STYLE_ROUNDED_GRID
#define TSTYLE_HEAVY_GRID tabulate::TABLE_STYLE_HEAVY_GRID
#define TSTYLE_MIXED_GRID tabulate::TABLE_STYLE_MIXED_GRID
#define TSTYLE_DOUBLE_GRID tabulate::TABLE_STYLE_DOUBLE_GRID
#define TSTYLE_FANCY_GRID tabulate::TABLE_STYLE_FANCY_GRID
#define TSTYLE_OUTLINE tabulate::TABLE_STYLE_OUTLINE
#define TSTYLE_SIMPLE_OUTLINE tabulate::TABLE_STYLE_SIMPLE_OUTLINE
#define TSTYLE_ROUNDED_OUTLINE tabulate::TABLE_STYLE_ROUNDED_OUTLINE
#define TSTYLE_HEAVY_OUTLINE tabulate::TABLE_STYLE_HEAVY_OUTLINE
#define TSTYLE_MIXED_OUTLINE tabulate::TABLE_STYLE_MIXED_OUTLINE
#define TSTYLE_DOUBLE_OUTLINE tabulate::TABLE_STYLE_DOUBLE_OUTLINE
#define TSTYLE_FANCY_OUTLINE tabulate::TABLE_STYLE_FANCY_OUTLINE
#define TALIGN_LEFT tabulate::TABLE_ALIGN_LEFT
#define TALIGN_RIGHT tabulate::TABLE_ALIGN_RIGHT
#define TALIGN_CENTER tabulate::TABLE_ALIGN_CENTER
#else
#define TSetStyle tabulate::tabulate.setDefaultStyle
#define TSetAlign tabulate::tabulate.setAlign
#define TCLEAR tabulate::tabulate.clear()
#define TH tabulate::tabulate.setHead()
#define TL tabulate::tabulate.setContent()
#define TABLE tabulate::tabulate
#define TENDL tabulate::tabulate.endLine()
#define TSHOW tabulate::tabulate.toString
#define TS_PLAIN tabulate::TABLE_STYLE_PLAIN
#define TS_SIMPLE tabulate::TABLE_STYLE_SIMPLE
#define TS_GRID tabulate::TABLE_STYLE_GRID
#define TS_SIMPLE_GRID tabulate::TABLE_STYLE_SIMPLE_GRID
#define TS_ROUNDED_GRID tabulate::TABLE_STYLE_ROUNDED_GRID
#define TS_HEAVY_GRID tabulate::TABLE_STYLE_HEAVY_GRID
#define TS_MIXED_GRID tabulate::TABLE_STYLE_MIXED_GRID
#define TS_DOUBLE_GRID tabulate::TABLE_STYLE_DOUBLE_GRID
#define TS_FANCY_GRID tabulate::TABLE_STYLE_FANCY_GRID
#define TS_OUTLINE tabulate::TABLE_STYLE_OUTLINE
#define TS_SIMPLE_OUTLINE tabulate::TABLE_STYLE_SIMPLE_OUTLINE
#define TS_ROUNDED_OUTLINE tabulate::TABLE_STYLE_ROUNDED_OUTLINE
#define TS_HEAVY_OUTLINE tabulate::TABLE_STYLE_HEAVY_OUTLINE
#define TS_MIXED_OUTLINE tabulate::TABLE_STYLE_MIXED_OUTLINE
#define TS_DOUBLE_OUTLINE tabulate::TABLE_STYLE_DOUBLE_OUTLINE
#define TS_FANCY_OUTLINE tabulate::TABLE_STYLE_FANCY_OUTLINE
#define TA_LEFT tabulate::TABLE_ALIGN_LEFT
#define TA_RIGHT tabulate::TABLE_ALIGN_RIGHT
#define TA_CENTER tabulate::TABLE_ALIGN_CENTER
#endif


namespace tabulate
{
    enum TableAlign
    {
        TABLE_ALIGN_LEFT,
        TABLE_ALIGN_RIGHT,
        TABLE_ALIGN_CENTER
    };

    enum TableStyle
    {
        TABLE_STYLE_PLAIN,
        TABLE_STYLE_SIMPLE,
        TABLE_STYLE_GRID,
        TABLE_STYLE_SIMPLE_GRID,
        TABLE_STYLE_ROUNDED_GRID,
        TABLE_STYLE_HEAVY_GRID,
        TABLE_STYLE_MIXED_GRID,
        TABLE_STYLE_DOUBLE_GRID,
        TABLE_STYLE_FANCY_GRID,
        TABLE_STYLE_OUTLINE,
        TABLE_STYLE_SIMPLE_OUTLINE,
        TABLE_STYLE_ROUNDED_OUTLINE,
        TABLE_STYLE_HEAVY_OUTLINE,
        TABLE_STYLE_MIXED_OUTLINE,
        TABLE_STYLE_DOUBLE_OUTLINE,
        TABLE_STYLE_FANCY_OUTLINE
    };

    class Table
    {
    public:

        enum Mode
        {
            TABLE_MODE_HEAD,
            TABLE_MODE_CONTENT,
            TABLE_MODE_UNCHOSE
        };

        enum Status
        {
            TABLE_STATUS_START,
            TABLE_STATUS_END,
            TABLE_STATUS_STOP
        };

        Table(){}

        Table& setHead(bool add=true)
        {
            mode_ = TABLE_MODE_HEAD;
            status_ = TABLE_STATUS_START;
            if (!add) head_.clear();
            return (*this);
        }

        Table& setContent()
        {
            mode_ = TABLE_MODE_CONTENT;
            status_ = TABLE_STATUS_START;
            one_content_.clear();
            return (*this);
        }
        
        template <class T>
        Table& operator<<(T value)
        {
            std::ostringstream oss;
            oss << value;
            
            if (status_ == TABLE_STATUS_START)
            {
                if (mode_ == TABLE_MODE_CONTENT)
                {
                    one_content_.push_back(oss.str());
                }
                else if (mode_ == TABLE_MODE_HEAD)
                {
                    head_.push_back(oss.str());
                    // std::cout<< head_.size() << std::endl;
                }
            }
            else if (status_ == TABLE_STATUS_END)
            {
                status_ = TABLE_STATUS_STOP;
                if (mode_ == TABLE_MODE_CONTENT)
                {
                    contents_.push_back(one_content_);
                    mode_ = TABLE_MODE_UNCHOSE;
                }
                else if (mode_ == TABLE_MODE_HEAD)
                {
                    mode_ = TABLE_MODE_UNCHOSE;
                }
            }
            return (*this);
        }

        std::string endLine()
        {
            // std::cout << "status start: " << (status_ == TABLE_STATUS_START) << std::endl;
            if (status_ == TABLE_STATUS_START)
            {
                status_ = TABLE_STATUS_END;
            }
            return "";
        }

        std::string toString()
        {
            return this->toString(default_style_);
        }

        std::string toString(TableStyle style) {
            std::ostringstream oss;
            std::vector<std::string> s = tableStyleMap_.at(style);

            // std::cout << head_.size() << std::endl;
            
            // 获取列的最大宽度
            std::vector<int> colWidths;
            for (const auto& col : head_) 
            {
                colWidths.push_back(col.size());
            }
            for (const auto& row : contents_) 
            {
                while(colWidths.size() < row.size())
                {
                    colWidths.push_back(0);
                }

                for (int i = 0; i < row.size(); ++i) 
                {
                    if (row[i].size() > colWidths[i]) 
                    {
                        colWidths[i] = row[i].size();
                    }
                }
            }

            while (align_.size()< colWidths.size())
            {
                align_.push_back(TABLE_ALIGN_LEFT);
            }
            
            oss << drawSeparator(s, colWidths, 0);
            // 绘制头部
            if (!head_.empty()) {
                while (head_.size() < colWidths.size())
                {
                    head_.push_back("");
                }
                oss << drawRow(s, head_, colWidths, true);
                oss << drawSeparator(s, colWidths, 1);
            }
 
            // 绘制内容
            for (int i=0;i<contents_.size();i++) {
                if (i) 
                {
                    oss << drawSeparator(s, colWidths, 2);
                }
                while (contents_[i].size() < colWidths.size())
                {
                    contents_[i].push_back("");
                }
                oss << drawRow(s, contents_[i], colWidths, false);
            }
            oss << drawSeparator(s, colWidths, 3);
 
            return oss.str();
        }

        void clear()
        {
            head_.clear();
            one_content_.clear();
            contents_.clear();
        }

        void setDefaultStyle(TableStyle default_style)
        {
            default_style_ = default_style;
        }

        void setAlign(std::vector<TableAlign> align)
        {
            align_ = align;
        }

    private:

        Mode mode_ = TABLE_MODE_UNCHOSE;
        Status status_ = TABLE_STATUS_STOP;
        TableStyle default_style_ = TABLE_STYLE_SIMPLE;
        std::vector<TableAlign> align_;

        std::vector<std::string> head_, one_content_;
        std::vector<std::vector<std::string>> contents_;

        // 左上角、上横边、上中角、右上角、标题外竖杠、标题内竖杠、标题左下角、标题横边、标题中下角，标题右下角、内容外竖杠、内容内竖杠、内容左角、内容边、内容中角、内容右角、内容左下角、内容下边、内容中下角、内容右下角
        const std::unordered_map<TableStyle, std::vector<std::string>> tableStyleMap_ = {
            {TABLE_STYLE_PLAIN, {"","","","","","","","","","","","","","","","","","","",""}},
            {TABLE_STYLE_SIMPLE, {"","","","","","  ","","-","  ","","","  ","","","","","","","",""}},
            {TABLE_STYLE_GRID, {"+","-","+","+","|","|","+","=","+","+","|","|","+","-","+","+","+","-","+","+"}},
            {TABLE_STYLE_SIMPLE_GRID, {"┌","─","┬","┐","│","│","├","─","┼","┤","│","│","├","─","┼","┤","└","─","┴","┘"}},
            {TABLE_STYLE_ROUNDED_GRID, {"╭","─","┬","╮","│","│","├","─","┼","┤","│","│","├","─","┼","┤","╰","─","┴","╯"}},
            {TABLE_STYLE_HEAVY_GRID, {"┏","━","┳","┓","┃","┃","┣","━","╋","┫","┃","┃","┣","━","╋","┫","┗","━","┻","┛"}},
            {TABLE_STYLE_MIXED_GRID, {"┍","━","┯","┑","│","│","┝","━","┿","┥","│","│","├","─","┼","┤","┕","━","┷","┙"}},
            {TABLE_STYLE_DOUBLE_GRID, {"╔","═","╦","╗","║","║","╠","═","╬","╣","║","║","╠","═","╬","╣","╚","═","╩","╝"}},
            {TABLE_STYLE_FANCY_GRID, {"╒","═","╤","╕","│","│","╞","═","╪","╡","│","│","├","─","┼","┤","╘","═","╧","╛"}},
            {TABLE_STYLE_OUTLINE, {"+","-","+","+","|","|","+","=","+","+","|","|","","","","","+","-","+","+"}},
            {TABLE_STYLE_SIMPLE_OUTLINE, {"┌","─","┬","┐","│","│","├","─","┼","┤","│","│","","","","","└","─","┴","┘"}},
            {TABLE_STYLE_ROUNDED_OUTLINE, {"╭","─","┬","╮","│","│","├","─","┼","┤","│","│","","","","","╰","─","┴","╯"}},
            {TABLE_STYLE_HEAVY_OUTLINE, {"┏","━","┳","┓","┃","┃","┣","━","╋","┫","┃","┃","","","","","┗","━","┻","┛"}},
            {TABLE_STYLE_MIXED_OUTLINE, {"┍","━","┯","┑","│","│","┝","━","┿","┥","│","│","","","","","┕","━","┷","┙"}},
            {TABLE_STYLE_DOUBLE_OUTLINE, {"╔","═","╦","╗","║","║","╠","═","╬","╣","║","║","","","","","╚","═","╩","╝"}},
            {TABLE_STYLE_FANCY_OUTLINE, {"╒","═","╤","╕","│","│","╞","═","╪","╡","│","│","","","","","╘","═","╧","╛"}}
        };

        std::string drawRow(const std::vector<std::string>& style, const std::vector<std::string>& row, const std::vector<int>& colWidths, bool isHeader) {
            std::ostringstream oss;
            int start_idx = isHeader?4:10;
            // 绘制左竖杠
            oss << style[start_idx];
            std::string middle_split = style[start_idx+1];

            // 绘制每个单元格
            for (int i = 0; i < row.size(); ++i) {
                // 填充空格以达到最大宽度
                if (i)
                {
                    oss << middle_split;
                }
                if (align_[i] == TABLE_ALIGN_LEFT)
                {
                    oss << " " << pystring(row[i]).ljust(colWidths[i]).str() << " ";
                }
                else if (align_[i] == TABLE_ALIGN_RIGHT)
                {
                    oss << " " << pystring(row[i]).rjust(colWidths[i]).str() << " ";
                }
                else if (align_[i] == TABLE_ALIGN_CENTER)
                {
                    oss << " " << pystring(row[i]).ljust((colWidths[i] + row[i].size())/2).rjust(colWidths[i]).str() << " ";
                }
            }
            // 绘制右竖杠
            oss << style[start_idx];
            // 每一行结束后添加换行符
            oss << std::endl;
            return oss.str();
        }
 
        std::string drawSeparator(const std::vector<std::string>& style, const std::vector<int>& colWidths, int flag) {
            std::ostringstream oss;

            int start_idx = (flag==3)?16:(flag * 6);

            // left split
            oss << style[start_idx];
            for (int i=0;i<colWidths.size();i++) 
            {
                if (i)
                {
                    // middle split
                    oss << style[start_idx+2];
                }
                oss << (pystring(style[start_idx+1])*(2+colWidths[i]));  // +2 是因为两边有边框字符
            }
            // right split
            oss << style[start_idx+3];
            if (oss.str().length() && flag != 3)
            {
                oss << std::endl;
            }

            return oss.str();
        }

    };

    static Table tabulate;
}

// static void test_table()
// {
//     // TCLEAR;
//     // TH << "Key"   << "Value" << TENDL;
//     // TL << "count" << 1       << TENDL;
//     // TL << "path" << "/home/lsh/code/cpp/perception_ros" << TENDL;

//     // std::cout << TSHOW(tabulate::TABLE_STYLE_FANCY_OUTLINE) << std::endl;

//     TCLEAR;

//     TH;
//     TABLE << "Key"   << "Value" << TENDL;

//     TL << "count" << 1       << TENDL;

//     TL;
//     TABLE << "path" << "/home/lsh/code/cpp/perception_ros";
//     TABLE << 1234567;
//     TABLE << TENDL;

//     TSetAlign({TA_LEFT, TA_CENTER});
//     for (int i=0;i<16;i++)
//     {
//         TSetStyle((tabulate::TableStyle)i);
//         std::cout << TSHOW() << std::endl;
//     }
// }

#endif

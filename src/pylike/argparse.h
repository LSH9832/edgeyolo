/**
 * Author: https://github.com/LSH9832
 * 感谢 https://github.com/0382, 本代码借鉴了 https://github.com/0382/util/blob/main/cpp/argparse/argparse.hpp 思路并完全重构，更接近于python的argparse用法
 */
#pragma once
#ifndef ARGPARSE_H
#define ARGPARSE_H

#include <iostream>
#include "./logger.h"
#include "./tabulate.h"


#define STORE_TRUE argparse::ACTION_OPT::ACTION_OPT_TRUE
#define STORE_FALSE argparse::ACTION_OPT::ACTION_OPT_FALSE

namespace argparse
{
    enum class ACTION_OPT
    {
        ACTION_OPT_TRUE,
        ACTION_OPT_FALSE
    };

    class OneArgument
    {
    public:
        OneArgument() {}
        OneArgument(pystring sname, pystring name, pystring value="", pystring help="", bool isRequired=false, bool isPosition=false, pystring nargs="1", bool isBool=false)
        {
            sname_ = sname;
            name_ = name;
            help_ = help;
            required = isRequired;
            nargs_ = nargs;
            if (!nargs_.isdigit())
            {
                bool flag = (nargs_ == "+") || (nargs_ == "*") || (nargs_ == "?");
                if (!flag)
                {
                    LERROR << "'" << name << "': nargs not valid!" << ENDL;
                    exit(-1);
                }
            }
            isBool_ = isBool;
            isPosition_ = isPosition;
            hasDefault = !value.empty();

            defaultValues_ = value.split(" ");
            isNargs_ = defaultValues_.size() > 1;
            
        }

        bool isValid()
        {
            return (values_.size() || defaultValues_.size());
        }

        bool isPosArg()
        {
            return isPosition_;
        }

        int numArgs()
        {
            if (nargs_.isdigit())
            {
                return nargs_;
            }
            return -1;
        }

        bool isRequired()
        {
            return required;
        }

        bool isRequiredButNotSet()
        {
            return required && useDefault;
        }

        void parseValue(pystring value="")
        {
            if (!value.empty())
            {
                values_ = value.split(" ");
                isNargs_ = values_.size() > 1;
                useDefault =false;
            }
        }

        size_t size()
        {
            return (useDefault?defaultValues_:values_).size();
        }

        OneArgument at(size_t idx)
        {
            if (!isNargs_)
            {
                LERROR << name_ << " is not iterable." << ENDL;
                exit(-1);
            }
            pystring subValue = (useDefault?defaultValues_:values_)[idx];
            return OneArgument("", name_ + "[" + std::to_string(idx) + "]", subValue, "", false, false);
        }

        OneArgument operator[](size_t idx)
        {
            return at(idx<0?(useDefault?defaultValues_:values_).size()+idx:idx);
        }

        template <class T>
        T as()
        {
            if (isNargs_ && (useDefault?defaultValues_:values_).size() > 1) 
            {
                LWARN << name_ << " is a list, use asList() instead of as(), return first value of the list" << ENDL;
            }
            return (T)((useDefault?defaultValues_:values_)[0]);
        }

        operator int() const {
            if (isNargs_ && (useDefault?defaultValues_:values_).size() > 1) 
            {
                LWARN << name_ << " is a list, use asList() instead of as(), return first value of the list" << ENDL;
            }
            return (int)((useDefault?defaultValues_:values_)[0]);
        }
    
        operator float() const {
            if (isNargs_ && (useDefault?defaultValues_:values_).size() > 1) 
            {
                LWARN << name_ << " is a list, use asList() instead of as(), return first value of the list" << ENDL;
            }
            return (float)((useDefault?defaultValues_:values_)[0]);
        }
    
        operator double() const {
            if (isNargs_ && (useDefault?defaultValues_:values_).size() > 1) 
            {
                LWARN << name_ << " is a list, use asList() instead of as(), return first value of the list" << ENDL;
            }
            return (double)((useDefault?defaultValues_:values_)[0]);
        }
    
        operator std::string() const {
            if (isNargs_ && (useDefault?defaultValues_:values_).size() > 1) 
            {
                LWARN << name_ << " is a list, use asList() instead of as(), return first value of the list" << ENDL;
            }
            return (std::string)((useDefault?defaultValues_:values_)[0]);
        }

        operator pystring() const {
            if (isNargs_ && (useDefault?defaultValues_:values_).size() > 1) 
            {
                LWARN << name_ << " is a list, use asList() instead of as(), return first value of the list" << ENDL;
            }
            return (pystring)((useDefault?defaultValues_:values_)[0]);
        }
    
        operator bool() const {
            if (isNargs_ && (useDefault?defaultValues_:values_).size() > 1) 
            {
                LWARN << name_ << " is a list, use asList() instead of as(), return first value of the list" << ENDL;
            }
            return (bool)((useDefault?defaultValues_:values_)[0]);
        }

        template <class T>
        std::vector<T> asList()
        {
            std::vector<T> rets;
            for (T ret: (useDefault?defaultValues_:values_))
            {
                rets.push_back(ret);
            }
            return rets;
        }

        pystring asString()
        {
            if (isNargs_)
            {
                std::ostringstream oss;
                oss << "[";
                for (pystring ret: (useDefault?defaultValues_:values_))
                {
                    oss << ret << ",";
                }
                pystring ret = oss.str();
                ret[-1] = ']';
                return ret;
            }
            return (useDefault?defaultValues_:values_)[0];
        }

        pystring sname_="", name_="", help_="", nargs_="";
        bool isBool_=false;

    private:
        bool required=false;
        bool hasDefault=false;
        bool useDefault=true;
        bool isNargs_=false;
        bool isPosition_=false;
        
        std::vector<pystring> values_={};

        std::vector<pystring> defaultValues_={};
    };

    class ArgumentParser
    {
    public:
        ArgumentParser(std::string description="", int argc=0, char** argv=nullptr)
        {
            description_ = description;
            argc_ = argc;
            argv_ = argv;
        }

        // void add_argument(pystring name, ACTION_OPT action, pystring help="")
        // {
        //     add_argument({name}, action, help);
        // }

        // void add_argument(pystring name, pystring default_value, pystring help="", pystring nargs="1")
        // {
        //     add_argument({name}, default_value, help, nargs);
        // }

        // template <class T>
        // void add_argument(pystring name, std::vector<T> default_value, pystring help="", pystring nargs="1")
        // {
        //     add_argument({name}, default_value, help, nargs);
        // }
        
        std::string showParams(bool showHelp=true, std::vector<pystring> longNames={})
        {
            if (longNames.empty())
            {
                longNames = this->names;
            }
#ifdef SAFE_DEFINE
            TableClear;
            TableHead << "Key" << "Value" << "Help" << TableEndLine;
            
            int count = 0;
            for(auto name: longNames)
            {
                if (!pyin(name, names)) continue;
                auto item = at(name);
                if (item.size() > 1)
                {
                    std::ostringstream oss;
                    oss << "[";
                    auto items = item.asList<std::string>();
                    for (int i=0;i<item.size();++i)
                    {
                        if (i) oss << ", ";
                        oss << items[i];
                    }
                    oss << "]";
                    TableLine << name << oss.str() << item.help_ << TableEndLine;
                }
                else
                {
                    TableLine << name << item.asString() << item.help_ << TableEndLine;
                }
            }
            TableSetAlign({TALIGN_LEFT, TALIGN_CENTER});
            TableSetStyle(TSTYLE_FANCY_GRID);
            return TableShow();
#else
            TCLEAR;
            if (showHelp)
            {
                TH << "Key" << "Value" << "Help" << TENDL;
            }
            else
            {
                TH << "Key" << "Value" << TENDL;
            }
            
            int count = 0;
            for(auto name: longNames)
            {
                if (!pyin(name, names)) continue;
                auto item = at(name);
                if (item.size() > 1)
                {
                    std::ostringstream oss;
                    oss << "[";
                    auto items = item.asList<std::string>();
                    for (int i=0;i<item.size();++i)
                    {
                        if (i) oss << ", ";
                        oss << items[i];
                    }
                    oss << "]";
                    if (showHelp)
                        TL << name << oss.str() << item.help_ << TENDL;
                    else
                        TL << name << oss.str() << TENDL;
                }
                else
                {
                    if (showHelp)
                        TL << name << item.asString() << item.help_ << TENDL;
                    else
                        TL << name << item.asString() << TENDL;
                }
            }
            TSetAlign({TA_LEFT, TA_CENTER, TA_LEFT});
            TSetStyle(TS_FANCY_GRID);
            return TSHOW();
#endif


        }

        // bool
        void add_argument(std::vector<pystring> names, ACTION_OPT action, pystring help="")
        {
            if (names.empty() || names.size() > 2)
            {
                LERROR << "number of argument name should be 1 or 2!" << ENDL;
                exit(-1);
            }

            pystring shortName="", longName = names[0];
            if (names.size() == 2)
            {
                shortName = names[0];
                longName = names[1];
            }

            bool isPos = checkName(shortName, longName);

            if (isPos)
            {
                LERROR << "bool argument should not be a position argument." << ENDL;
                exit(-1);
            }

            pystring name = longName.substr(2, longName.length());
            args_.push_back(OneArgument(shortName, longName, (action==ACTION_OPT::ACTION_OPT_TRUE)?"false":"true", help, false, false, "1", true));

            if (!shortName.empty()) 
            {
                snames.push_back(shortName);
                maxSlen = (shortName.length()>maxSlen)?shortName.length():maxSlen;
            }
            maxLlen = (longName.length()>maxLlen)?longName.length():maxLlen;
            this->names.push_back(name);
        }

        void add_argument(std::vector<pystring> names, pystring default_value, pystring help="", pystring nargs="1")
        {
            if (default_value == "true" || default_value == "false")
            {
                add_argument_required(names, default_value, help, nargs);
                return;
            }
            if (names.empty() || names.size() > 2)
            {
                LERROR << "number of argument name should be 1 or 2!" << ENDL;
                exit(-1);
            }

            pystring shortName="", longName = names[0];
            if (names.size() == 2)
            {
                shortName = names[0];
                longName = names[1];
            }
            bool isPos = checkName(shortName, longName);
            if (isPos)
            {
                LERROR << "position argument should not set default value." << ENDL;
                exit(-1);
            }
            pystring name = isPos?longName:longName.substr(2, longName.length());

            args_.push_back(OneArgument(shortName, longName, default_value, help, false, isPos, nargs));

            if (!shortName.empty()) 
            {
                snames.push_back(shortName);
                maxSlen = (shortName.length()>maxSlen)?shortName.length():maxSlen;
            }
            maxLlen = (longName.length()>maxLlen)?longName.length():maxLlen;
            this->names.push_back(name);
        }
        
        // if no default value, you should inform whether this argument is required.
        void add_argument_required(std::vector<pystring> names, bool required, pystring help="", pystring nargs="1")
        {
            if (names.empty() || names.size() > 2)
            {
                LERROR << "number of argument name should be 1 or 2!" << ENDL;
                exit(-1);
            }

            pystring shortName="", longName = names[0];
            if (names.size() == 2)
            {
                shortName = names[0];
                longName = names[1];
            }
            bool isPos = checkName(shortName, longName);
            
            pystring name = isPos?longName:longName.substr(2, longName.length());
            if (isPos)
            {
                numPosArgs++;
                posNames.push_back(name);
            }
            args_.push_back(OneArgument(shortName, longName, "", help, isPos?true:required, isPos, nargs));

            if (!shortName.empty()) 
            {
                snames.push_back(shortName);
                maxSlen = (shortName.length()>maxSlen)?shortName.length():maxSlen;
            }
            maxLlen = (longName.length()>maxLlen)?longName.length():maxLlen;
            this->names.push_back(name);
        }

        template <class T>
        void add_argument(std::vector<pystring> names, std::vector<T> default_value, pystring help="", pystring nargs="1")
        {
            if (names.empty() || names.size() > 2)
            {
                LERROR << "number of argument name should be 1 or 2!" << ENDL;
                exit(-1);
            }

            pystring shortName="", longName = names[0];
            if (names.size() == 2)
            {
                shortName = names[0];
                longName = names[1];
            }
            bool isPos = checkName(shortName, longName);
            pystring name = isPos?longName:longName.substr(2, longName.length());

            if (isPos)
            {
                LERROR << "position argument should not set default value." << ENDL;
                exit(-1);
            }

            std::ostringstream oss;
            for (int i=0;i<default_value.size();i++)
            {
                if (i) oss << " ";
                oss << default_value[i];
            }
            args_.push_back(OneArgument(shortName, longName, oss.str(), help, false, isPos, nargs));

            if (!shortName.empty()) 
            {
                snames.push_back(shortName);
                maxSlen = (shortName.length()>maxSlen)?shortName.length():maxSlen;
            }
            maxLlen = (longName.length()>maxLlen)?longName.length():maxLlen;
            this->names.push_back(name);
        }

        ArgumentParser parse_args(bool add_log=false)
        {
            // std::cout << names.size() << ", " << args_.size() << std::endl;
            checkHelp();
            int idx = 1;
            pystring currentName = "";
            while (idx < argc_)
            {
                // std::cout << idx << "/" << argc_ << std::endl;
                pystring arg = argv_[idx];

                // INFO << arg << "," << numPosArgs << ENDL;

                if (idx < numPosArgs+1)
                {
                    if (arg.startswith("-"))
                    {
                        LERROR << "parse error" << ENDL;
                        showHelpMsg();
                        exit(-1);
                    }
                    
                    // 
                    at(posNames[idx-1]).parseValue(arg);
                }
                else
                {
                    // INFO << "opt arg" << ENDL;
                    if (!arg.startswith("-"))
                    {
                        LERROR << "parse error at '" << arg << "'." << ENDL;
                        showHelpMsg();
                        exit(-1);
                    }

                    if (arg.startswith("--"))
                    {
                        // INFO << "long name" << ENDL;
                        currentName = arg.substr(2, arg.length());
                        bool existName = false;
                        for (pystring name: names)
                        {
                            if (name == currentName)
                            {
                                existName = true;
                                break;
                            }
                        }
                        if (!existName)
                        {
                            LERROR << "error: arg '" << arg << "' not exist!" << ENDL;
                            showHelpMsg();
                            exit(-1);
                        }
                    }
                    else
                    {
                        // INFO << "short name" << ENDL;
                        bool existSName = false;
                        for (pystring sname: snames)
                        {
                            if (sname == arg)
                            {
                                existSName = true;
                                break;
                            }
                        }
                        if (!existSName)
                        {
                            LERROR << "error: arg '" << arg << "' not exist!" << ENDL;
                            showHelpMsg();
                            exit(-1);
                        }
                        // INFO << "finding arg info" << ENDL;
                        for (OneArgument nowarg: args_)
                        {
                            if (nowarg.sname_ == arg)
                            {
                                currentName = nowarg.name_;
                                if (!nowarg.isPosArg())
                                {
                                    currentName = currentName.substr(2, currentName.length());
                                }
                                else
                                {
                                    LERROR << "positional argument should not have short name" << ENDL;
                                    exit(-1);
                                }
                                break;
                            }
                        }
                        // INFO << "finding arg info end" << ENDL;

                    }

                    if (at(currentName).isBool_)
                    {
                        at(currentName).parseValue(!at(currentName).as<bool>()); // 取反
                    }
                    else
                    {
                        // 如果不为bool，则arg不能是最后一个位置，下一个位置也不能是下一个arg
                        if (idx == argc_)
                        {
                            LERROR << "arg '" << arg << "' has no value!" << ENDL;
                            exit(-1);
                        }
                        if (pystring(argv_[idx+1]).startswith("-"))
                        {
                            LERROR << "arg '" << arg << "' has no value!" << ENDL;
                            exit(-1);
                        }
                        // 直到下一个arg之前的所有值
                        int validNum = at(currentName).numArgs();
                        std::ostringstream oss;
                        int count = 0;
                        while (idx < argc_-1)
                        {
                            idx++;
                            pystring nowStr = argv_[idx];
                            if (nowStr.startswith("-"))
                            {
                                idx--;
                                break;
                            }
                            if (count == validNum) continue;
                            if (count) oss << " ";
                            oss << nowStr;
                            count++;
                        }
                        at(currentName).parseValue(oss.str());

                    }
                }
                idx++;
            }

            for(pystring name: names)
            {
                if (at(name).isRequiredButNotSet())
                {
                    LERROR << "arg '" << name << "' is required." << ENDL;
                    showHelpMsg();
                    exit(-1);
                }
            }
            return *this;
        }

        OneArgument& at(pystring name){
            if (!pyin(name, names))
            {
                // ERROR << "no arg named " << name << "!" << ENDL;
                // exit(-1);
                // auto ret = OneArgument();
                return unknown_arg;
            }
            for (int i=0;i<args_.size();i++)
            {
                if(args_[i].isPosArg())
                {
                    if (args_[i].name_ == name)
                    {
                        return args_[i];
                    }
                }
                else
                {
                    if (args_[i].name_.substr(2, args_[i].name_.length()) == name)
                    {
                        return args_[i];
                    }
                }
            }

            return unknown_arg;
        }
        
        OneArgument& operator[](pystring name){
            return at(name);
        }
    
    private:

        void checkHelp()
        {
            int idx = 0;
            while (idx < argc_)
            {
                pystring arg = argv_[idx];
                if (arg == "-?" || arg == "--help")
                {
                    showHelpMsg();
                    exit(0);
                }
                else if (arg == "--?")
                {
                    showHelpMsg(true);
                    exit(0);
                }
                
                idx++;
            }
        }    


        bool checkName(pystring shortName, pystring longName)
        {

            bool svalid = shortName.empty() || (shortName.length()>1 && shortName.startswith("-") && (!shortName.startswith("--")));
            bool lvalid = longName.length()>2 && longName.startswith("--");

            if (svalid && lvalid)
            {
                if (!shortName.empty())
                {
                    for (pystring sname: snames)
                    {
                        if (sname == shortName)
                        {
                            LERROR << "short name '" << sname << "' already exist!" << ENDL;
                            exit(-1);
                        }
                    }
                }
                for (pystring name: names)
                {
                    if (longName.substr(2, longName.length()) == name)
                    {
                        LERROR << "name '" << name << "' already exist!" << ENDL;
                        exit(-1);
                    }
                }

                return false;
            }
            else
            {
                svalid = shortName.empty() || (shortName.length() && !shortName.startswith("-"));
                lvalid = longName.length() && !longName.startswith("-");

                

                if (svalid && lvalid)
                {
                    for (pystring name: names)
                    {
                        if (longName == name)
                        {
                            LERROR << "name '" << name << "' already exist!" << ENDL;
                            exit(-1);
                        }
                    }
                    return true;
                }
                else
                {
                    LERROR << "short name or long name not valid: [" << shortName << ", " << longName << "]" << ENDL;
                    exit(-1);
                }
            }
            
        }

        void showHelpMsg(bool simple=false)
        {
            std::vector<pystring> posUsages, optUsages;
            std::vector<pystring> posHelps, optHelps;

            std::ostringstream oneLineHelp, posOneLine, optOneLine, simpleHelp;
            oneLineHelp << "usage: " << argv_[0] << " [-?]";
            // optOneLine ;
            
            int maxUsageLen=0;
            int count = 0;
            for (auto name: names)
            {
                if (count++)
                {
                    simpleHelp << "\n";
                }
                std::ostringstream thisUsage;
                auto arg = at(name);
                if (arg.isPosArg())
                {
                    posOneLine << " [" << name << "]";
                    thisUsage << "  " << name;
                    simpleHelp << name;

                    posUsages.push_back(thisUsage.str());
                    posHelps.push_back(arg.help_);
                }
                else
                {
                    optOneLine << " [";
                    if (arg.sname_.length())
                    {
                        optOneLine << arg.sname_;
                    }
                    else
                    {
                        optOneLine << arg.name_;
                    }
                    simpleHelp << arg.name_;

                    if(!arg.isBool_)
                    {
                        optOneLine << " [" << name.upper().replace("-", "_");
                        if (arg.numArgs() != 1)
                        {
                            optOneLine << " ...";
                            simpleHelp << " []";
                        }
                        else
                        {
                            simpleHelp << " " << (arg.asString().length()?arg.asString():"\"\"");
                        }
                        optOneLine << "]";
                    }
                    else
                    {
                        simpleHelp << " false";
                    }
                    optOneLine << "]";

                    // 
                    thisUsage << "  ";
                    if (arg.sname_.length())
                    {
                        thisUsage << arg.sname_ << ", ";
                    }
                    thisUsage << arg.name_;
                    if(!arg.isBool_)
                    {
                        thisUsage << " [" << name.upper().replace("-", "_");
                        if (arg.numArgs() != 1)
                        {
                            thisUsage << " ...";
                        }
                        thisUsage << "]";
                    }
                    optUsages.push_back(thisUsage.str());
                    optHelps.push_back(arg.help_);
                }
                
                int thisLength = thisUsage.str().length();
                maxUsageLen = MAX(thisLength, maxUsageLen);
            }
            maxUsageLen += 8;

            std::ostringstream multiLineHelp, posMultiLine, optMultiLine;
            optMultiLine << "optional arguments:\n";
            optMultiLine << pystring("  -?, --help").ljust(maxUsageLen) << "show this help message and exit\n";
            for (int i=0;i<optUsages.size();i++)
            {
                optMultiLine << optUsages[i].ljust(maxUsageLen) << optHelps[i] << "\n";
            }

            posMultiLine << "positional arguments:\n";
            for (int i=0;i<posUsages.size();i++)
            {
                posMultiLine << posUsages[i].ljust(maxUsageLen) << posHelps[i] << "\n";
            }

            oneLineHelp << posOneLine.str() << optOneLine.str();
            if (posUsages.size())
            {
                multiLineHelp << posMultiLine.str() << "\n";
            }
            multiLineHelp << optMultiLine.str();
            
            if (simple) 
            {
                std::cout << simpleHelp.str() << std::endl;
            }
            else
            {
                std::cout << description_ << (description_.empty()?"":"\n") << oneLineHelp.str() << "\n\n" << multiLineHelp.str() << std::endl;
            }
            

            // exit(0);
        }

        int argc_ = 0;
        char** argv_ = nullptr;

        int numPosArgs=0;

        pystring description_="";

        std::vector<pystring> snames, names;
        std::vector<pystring> posNames;

        std::vector<OneArgument> args_;
        OneArgument unknown_arg;
        int maxSlen=0, maxLlen=0;
        
    };


    
}

#endif

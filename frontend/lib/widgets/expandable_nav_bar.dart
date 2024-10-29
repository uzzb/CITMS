import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class ExpandableNavBar extends StatefulWidget {
  final int selectedIndex;
  final ValueChanged<int> onItemSelected;

  ExpandableNavBar({required this.selectedIndex, required this.onItemSelected});

  @override
  _ExpandableNavBarState createState() => _ExpandableNavBarState();
}

class _ExpandableNavBarState extends State<ExpandableNavBar> {
  bool isExpanded = false;


  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: Duration(milliseconds: 300),
      width: isExpanded ? 200 : 60,
      height: 60,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(30),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          _buildNavItem(0, 'assets/images/align-justify.svg'),
          if (isExpanded)
            Expanded (
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Flexible(child: _buildNavItem(1, 'assets/images/monitor.svg')),
                  // Flexible(child:_buildNavItem(2, 'assets/images/data_analysis.svg')),
                  Flexible(child:_buildNavItem(2, 'assets/images/user.svg')),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildNavItem(int index, String iconPath) {
    bool isSelected = widget.selectedIndex == index;
    bool isMenuIcon = index == 0;
    Color backgroundColor = isMenuIcon
      ? (isExpanded ? Colors.white : Color(0xFFFB1515))
      : (isSelected ? Color(0xFFFB1515) : Colors.transparent);
    Color iconColor = isMenuIcon
      ? (isExpanded ? Color(0xFFFB1515) : Colors.white)
      : (isSelected ? Colors.white : Colors.black);

    return GestureDetector(
      onTap: () {
        if (isMenuIcon) {
          setState(() {
            isExpanded = !isExpanded;
          });
        } else {
          widget.onItemSelected(index);
        }
      },
      child: Container(
        width: 40,
        height: 40,
        margin: EdgeInsets.symmetric(horizontal: 10),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: backgroundColor,
        ),
        child: Center(
          child: SvgPicture.asset(
            iconPath,
            width: 24,
            height: 24,
            colorFilter: ColorFilter.mode(
              iconColor,
              BlendMode.srcIn,
            ),
          ),
        ),
      ),
    );
  }

}
